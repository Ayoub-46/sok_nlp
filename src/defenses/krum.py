import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np

from ..fl.server import FedAvgServer
from .metrics_mixin import DefenseMetricsMixin

class MKrumServer(DefenseMetricsMixin, FedAvgServer):
    """
    Implements the Multi-Krum (M-Krum) defense mechanism.
    """
    def __init__(self, *args, **kwargs):
        # Initialize DefenseMetricsMixin first
        DefenseMetricsMixin.__init__(self, *args, **kwargs)
        # Initialize FedAvgServer
        FedAvgServer.__init__(self, *args, **kwargs)
        
        # Defense Config
        config = kwargs.get('defense_config', {})
        self.num_byzantine = config.get('krum_f', 0)
        self.num_to_select = config.get('krum_m', 1)
        
        print(f"Initialized MKrumServer: f={self.num_byzantine}, m={self.num_to_select}")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        num_updates = len(self.received_updates)
        client_ids_list = list(self.received_updates.keys())
        
        if num_updates == 0:
            return self.get_params()

        # Check condition n > 2f + 2
        if num_updates <= 2 * self.num_byzantine + 2:
            print(f"Warning: Not enough clients ({num_updates}) for Krum (f={self.num_byzantine}). Standard Avg.")
            
            # Log metrics as "All Accepted"
            self.update_defense_metrics(
                client_ids_received=set(client_ids_list),
                rejected_client_ids=set() 
            )
            return super().aggregate()
        
        # 1. Get deltas
        global_params = self.get_params()
        client_deltas = []
        for cid in client_ids_list:
            local_params = self.received_updates[cid]['params']
            delta = {k: local_params[k] - global_params[k] for k in local_params}
            client_deltas.append(delta)

        # 2. Flatten for distance
        flat_deltas = [torch.cat([p.flatten() for p in d.values()]) for d in client_deltas]

        # 3. Pairwise Distances
        distances = torch.zeros((num_updates, num_updates))
        for i in range(num_updates):
            for j in range(i, num_updates):
                dist = torch.linalg.norm(flat_deltas[i] - flat_deltas[j]) ** 2
                distances[i, j] = distances[j, i] = dist.item()

        # 4. Krum Scores (Sum of k nearest distances)
        scores = []
        num_neighbors = num_updates - self.num_byzantine - 2
        for i in range(num_updates):
            sorted_dists, _ = torch.sort(distances[i])
            # Exclude self (index 0) and take next 'num_neighbors'
            scores.append(torch.sum(sorted_dists[1:num_neighbors+1]).item())
        
        # 5. Select Top-m lowest scores
        sorted_indices = np.argsort(scores)
        selected_indices = sorted_indices[:self.num_to_select]
        selected_client_ids = [client_ids_list[i] for i in selected_indices]
        
        print(f"Krum selected: {selected_client_ids}")

        # 6. Metrics Update
        self.update_defense_metrics(
            client_ids_received=set(client_ids_list),
            rejected_client_ids=set(client_ids_list) - set(selected_client_ids)
        )

        # 7. Aggregate Selection
        total_samples = sum(self.received_updates[cid]['samples'] for cid in selected_client_ids)
        
        if total_samples == 0:
            self.received_updates = {}
            return self.get_params()

        averaged = {}
        first_weights = self.received_updates[selected_client_ids[0]]['params']
        for k in first_weights.keys():
            averaged[k] = torch.zeros_like(first_weights[k])
            
        for cid in selected_client_ids:
            weight = self.received_updates[cid]['samples'] / total_samples
            w_k = self.received_updates[cid]['params']
            for k in averaged:
                averaged[k] += w_k[k] * weight

        # Apply to global model
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})
        self.received_updates = {}
        
        return {k: v.cpu().clone() for k, v in averaged.items()}