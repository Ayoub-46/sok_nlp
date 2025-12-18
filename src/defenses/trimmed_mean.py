import torch
from typing import Dict
from ..fl.server import FedAvgServer
from .metrics_mixin import DefenseMetricsMixin

class TrimmedMeanServer(DefenseMetricsMixin, FedAvgServer):
    def __init__(self, *args, **kwargs):
        # Initialize both parents
        super().__init__(*args, **kwargs)
        config = kwargs.get('defense_cfg')
        if config is None and len(args) >= 4:
            config = args[3]
        
        self.config = config if config is not None else {}
        self.beta = self.config.get('beta', 0.1)  # Fraction to trim
        self.round_counter = 0
        print(f"Initialized TrimmedMeanServer with beta={self.beta}.")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        if not self.received_updates: return self.get_params()
        
        # 1. Setup
        client_ids = list(self.received_updates.keys())
        first_id = client_ids[0]
        num_clients = len(client_ids)
        num_trim = int(self.beta * num_clients)
        
        client_trim_counts = torch.zeros(num_clients, dtype=torch.long, device=self.device)
        total_params_count = 0

        averaged_params = {}
        
        # 2. Iterate Parameters
        for name in self.received_updates[first_id]['params'].keys():
            stacked = torch.stack([self.received_updates[cid]['params'][name] for cid in client_ids]).to(self.device)
            
            if 'num_batches_tracked' in name:
                averaged_params[name] = stacked[0] # Aggregation logic for buffers
                continue
            
            # Sort along clients (dim=0) to find outliers
            # indices: (Num_Clients, ...) tells us which client index is at which rank
            sorted_stack, indices = torch.sort(stacked, dim=0)
            
            # 3. Trim Logic
            if 2 * num_trim < num_clients:
                trimmed_stack = sorted_stack[num_trim : num_clients - num_trim]
                
                rejected_indices = torch.cat((indices[:num_trim], indices[num_clients-num_trim:]), dim=0)
                
                counts = torch.bincount(rejected_indices.flatten(), minlength=num_clients)
                client_trim_counts += counts
                
                total_params_count += stacked[0].numel()
            else:
                trimmed_stack = sorted_stack 
            
            averaged_params[name] = torch.mean(trimmed_stack, dim=0)

        # 4. Finalize Metrics (Heuristic: Reject if trimmed in > 50% of parameters)
        rejected_client_ids = set()
        
        if total_params_count > 0:
            trim_ratios = (client_trim_counts.float() / total_params_count).cpu().numpy()
            
            for idx, ratio in enumerate(trim_ratios):
                if ratio > 0.5: 
                    rejected_client_ids.add(client_ids[idx])

        self.update_defense_metrics(set(client_ids), rejected_client_ids)
        
        # 5. Update Server Model
        self.set_params({k: v.to(self.device) for k, v in averaged_params.items()})
        self.received_updates = {}
        
        return averaged_params


# MedianServer Implementation (incorporating DefenseMetricsMixin)
class MedianServer(DefenseMetricsMixin, FedAvgServer):
    """Coordinate-wise Median Aggregation with TPR/FPR logging."""
    def __init__(self, *args, **kwargs):
        # Initialize both parents
        super().__init__(*args, **kwargs)
        
        self.round_counter = 0
             
    def aggregate(self) -> Dict[str, torch.Tensor]:
        if not self.received_updates: return self.get_params()
        
        client_ids = list(self.received_updates.keys())
        first_id = client_ids[0]
        num_clients = len(client_ids)
        
        # Track how many coordinates *did not* match the median for each client
        client_discard_counts = torch.zeros(num_clients, dtype=torch.long, device=self.device)
        total_params_count = 0
        
        averaged_params = {}
        
        # 1. Iterate over one client's keys to get parameter names
        for name in self.received_updates[first_id]['params'].keys():
            # Stack all client updates for this parameter: (Num_Clients, Param_Shape)
            stacked = torch.stack([self.received_updates[cid]['params'][name] for cid in client_ids]).to(self.device)
            
            if 'num_batches_tracked' in name:
                averaged_params[name] = stacked[0] # Aggregation logic for buffers
                continue
            
            # Compute median along dimension 0 (clients)
            median_val, _ = torch.median(stacked, dim=0)
            averaged_params[name] = median_val
            
            # --- METRIC TRACKING ---
            # Create a stacked tensor of the median value (Num_Clients, Param_Shape)
            median_val_expanded = median_val.unsqueeze(0).expand_as(stacked)
            
            # Mask where client update DOES NOT equal the aggregated median update (True=Discarded/Outlier)
            # Use strict comparison as a proxy for "not chosen as the median value"
            is_discarded = (stacked != median_val_expanded)
            
            # Sum discarded coordinates for each client (Num_Clients,)
            # The sum over dimensions 1..N collapses the shape to (Num_Clients,)
            discard_count_per_client = is_discarded.sum(dim=tuple(range(1, is_discarded.ndim)))
            
            client_discard_counts += discard_count_per_client.long()
            total_params_count += stacked[0].numel()
            # -----------------------

        # 2. Finalize Metrics (Heuristic: Reject if discarded in > 50% of parameters)
        rejected_client_ids = set()
        
        if total_params_count > 0:
            discard_ratios = (client_discard_counts.float() / total_params_count).cpu().numpy()
            
            for idx, ratio in enumerate(discard_ratios):
                # If client is discarded across a majority of coordinates, mark as rejected.
                if ratio > 0.5: 
                    rejected_client_ids.add(client_ids[idx])

        # Call the mixin to log TPR/FPR
        self.update_defense_metrics(set(client_ids), rejected_client_ids)
         # Increment round counter
        
        # 3. Update Server Model
        self.set_params({k: v.to(self.device) for k, v in averaged_params.items()})
        self.received_updates = {}
        return averaged_params