import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy

# Flame requires hdbscan for clustering
try:
    import hdbscan
except ImportError:
    print("Warning: hdbscan not installed. FLAME will fail. Run: pip install hdbscan")
    hdbscan = None

from ..fl.server import FedAvgServer
from .metrics_mixin import DefenseMetricsMixin

class FlameServer(DefenseMetricsMixin, FedAvgServer):
    """
    Implements the FLAME defense mechanism for NLP.
    1. Clustering (HDBSCAN) on last-layer gradients to identify clusters.
    2. Adaptive Clipping based on the median distance of benign clients.
    3. Noise Injection to mask remaining backdoor traces.
    """
    def __init__(self, *args, **kwargs):
        # Explicitly initialize both parents
        DefenseMetricsMixin.__init__(self, *args, **kwargs)
        FedAvgServer.__init__(self, *args, **kwargs)

        if hdbscan is None:
            raise ImportError("hdbscan is not installed. Please install it (pip install hdbscan) to use FlameServer.")

        # Extract defense config
        config = kwargs.get('defense_config')
        if config is None and len(args) >= 4:
            config = args[3]

        self.config = config if config is not None else {}
        self.lamda = self.config.get('flame_lamda', 0.001) # Noise scaler
        self.eta = self.config.get('flame_eta', 1.0)       # Global Learning Rate (Server LR)
        
        print(f"Initialized FlameServer with lamda={self.lamda}, eta={self.eta}")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        if not self.received_updates:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        client_ids_received_list = list(self.received_updates.keys())
        
        # 1. Detect anomalies using Clustering
        benign_client_ids, malicious_client_ids, client_distances = self.detect_anomalies()
        
        if malicious_client_ids:
            print(f"FLAME detected and rejected {len(malicious_client_ids)} clients: {malicious_client_ids}")

        # Log Metrics
        self.update_defense_metrics(
            client_ids_received=set(client_ids_received_list),
            rejected_client_ids=set(malicious_client_ids)
        )

        if not benign_client_ids:
            print("Warning: FLAME filtered out ALL clients. Global model not updated.")
            self.received_updates = {}
            return self.get_params()

        # 2. Robust Aggregation: Clipping + Noise
        
        # Calculate clipping norm (median Euclidean distance of benign updates)
        benign_distances = [client_distances[cid] for cid in benign_client_ids]
        clip_norm = torch.median(torch.tensor(benign_distances)).item() if benign_distances else 1.0

        # Accumulator for the global update
        weight_accumulator = {
            name: torch.zeros_like(param).to(self.device) 
            for name, param in self.global_model.named_parameters()
        }
        
        global_params_cpu = self.get_params() 

        for client_id in benign_client_ids:
            local_params = self.received_updates[client_id]['params']
            
            # FLAME uses unweighted averaging among the benign set to prevent 
            # attackers from claiming high sample counts to skew the model.
            weight = 1.0 / len(benign_client_ids) 

            for name, param_cpu in local_params.items():
                if name.endswith('num_batches_tracked') or name not in weight_accumulator: 
                    continue
                
                # Calculate Delta: (Local - Global)
                # Move to GPU for calculation
                diff = param_cpu.to(self.device) - global_params_cpu[name].to(self.device)
                
                # Apply Dynamic Clipping
                client_dist = client_distances[client_id]
                scaling_factor = 1.0
                if client_dist > clip_norm:
                    scaling_factor = clip_norm / client_dist
                
                weight_accumulator[name].add_(diff * scaling_factor * weight)

        # 3. Apply Update to Global Model
        # w_new = w_old + eta * accumulated_update + noise
        
        final_state_dict = self.global_model.state_dict()
        std_dev = self.lamda * clip_norm 

        for name, param in final_state_dict.items():
            if name in weight_accumulator:
                # Add aggregated update
                param.data.add_(weight_accumulator[name] * self.eta)

                # Add Adaptive Noise (only to weights/biases, not stats)
                if 'weight' in name or 'bias' in name:
                    noise = torch.normal(0, std_dev, param.shape, device=self.device)
                    param.data.add_(noise)
        
        # 4. Cleanup
        self.received_updates = {} 
        
        # Return updated parameters (CPU)
        return self.get_params()

    def detect_anomalies(self) -> Tuple[List[int], List[int], Dict[int, float]]:
        """
        Clustering on Last-Layer Gradients (Cosine Similarity).
        """
        num_clients = len(self.received_updates)
        client_ids_list = list(self.received_updates.keys())
        index_to_id = {i: cid for i, cid in enumerate(client_ids_list)}

        if num_clients < 2:
            return client_ids_list, [], {cid: 0.0 for cid in client_ids_list}

        global_params_cpu = self.get_params()
        
        # Identify last layer
        first_client_data = next(iter(self.received_updates.values()))
        last_layer_names = self._get_last_layers(first_client_data['params'])

        all_client_weights_for_clustering = []
        client_id_to_distance: Dict[int, float] = {}

        for client_id in client_ids_list:
            local_params = self.received_updates[client_id]['params']
            
            flat_update_diff = []
            last_layer_weights = []

            for name, param in local_params.items():
                # For distance calculation (Clipping): Use ALL weights
                if 'weight' in name or 'bias' in name:
                    diff = param.to(self.device) - global_params_cpu[name].to(self.device)
                    flat_update_diff.append(diff.flatten())
                
                # For Clustering: Use ONLY Last Layer
                if name in last_layer_names:
                    # Note: Using raw weights for clustering is standard in FLAME paper,
                    # though some versions use gradients. Here we use raw weights 
                    # (effectively gradients if we consider shift from global).
                    # Actually, let's use the DELTA of the last layer to be precise.
                    delta = param.to(self.device) - global_params_cpu[name].to(self.device)
                    last_layer_weights.append(delta.cpu().flatten())

            # Metric 1: Euclidean Distance (for Clipping)
            euclidean_dist = torch.linalg.norm(torch.cat(flat_update_diff)).item()
            client_id_to_distance[client_id] = euclidean_dist
            
            # Metric 2: Last Layer Delta (for Clustering)
            all_client_weights_for_clustering.append(
                torch.cat(last_layer_weights).numpy().astype(np.float64)
            )
        
        # HDBSCAN Clustering
        client_weights_array = np.array(all_client_weights_for_clustering, dtype=np.float64)

        clusterer = hdbscan.HDBSCAN(
            metric="cosine", 
            algorithm="generic",
            min_cluster_size=max(2, num_clients // 2 + 1), # Assumption: >50% are benign
            allow_single_cluster=True
        )
        
        labels = clusterer.fit_predict(client_weights_array)

        # Identify Benign Cluster
        benign_indices = []
        
        # Case A: Noise (-1) or Single Cluster -> Treat all as benign (FLAME logic)
        # Note: If -1 is dominant, it implies high variance, but we default to accept.
        if np.all(labels == -1) or len(np.unique(labels)) == 1:
            benign_indices = list(range(num_clients))
        else:
            # Case B: Multiple clusters -> Pick largest non-noise cluster
            # Exclude noise label -1
            valid_labels = labels[labels != -1]
            if len(valid_labels) > 0:
                unique_labels, counts = np.unique(valid_labels, return_counts=True)
                largest_cluster_label = unique_labels[np.argmax(counts)]
                benign_indices = [i for i, label in enumerate(labels) if label == largest_cluster_label]
            else:
                # Only noise found
                benign_indices = list(range(num_clients))

        benign_client_ids = [index_to_id[i] for i in benign_indices]
        malicious_client_ids = list(set(client_ids_list) - set(benign_client_ids))

        return benign_client_ids, malicious_client_ids, client_id_to_distance

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """Get names of last two layers with parameters (Weight + Bias)."""
        layer_names = list(state_dict.keys())
        param_layers = [name for name in layer_names if 'weight' in name or 'bias' in name]
        return param_layers[-2:]