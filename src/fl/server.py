from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import copy 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaseServer(ABC):
    @abstractmethod
    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def aggregate(self) -> Dict[str, torch.Tensor]:
        pass

class FedAvgServer(BaseServer):
    """
    Standard Federated Averaging (FedAvg) Server.
    Now supports keying updates by Client ID.
    """
    def __init__(self, 
                 global_model: nn.Module, 
                 device: str = "cpu",
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 **kwargs):
        
        self.global_model = global_model
        self.device = device
        self.criterion = criterion
        
        # Buffer to store client updates: {client_id: {'params': state_dict, 'samples': num_samples}}
        self.received_updates: Dict[int, Dict[str, Any]] = {}

    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.global_model.load_state_dict(state_dict)

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Returns CPU-based state dict to send to clients."""
        self.global_model.cpu()
        return copy.deepcopy(self.global_model.state_dict())

    def receive_update(self, client_id: int, state_dict: Dict[str, torch.Tensor], num_samples: int) -> None:
        """
        Collects a client's update. Now requires client_id.
        """
        self.received_updates[client_id] = {
            'params': copy.deepcopy(state_dict),
            'samples': num_samples
        }

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Performs FedAvg: w_global = sum(w_k * n_k) / sum(n_k)
        """
        if not self.received_updates:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        # 1. Calculate total samples
        total_samples = sum(info['samples'] for info in self.received_updates.values())
        
        # 2. Initialize aggregated weights
        # Get first client's params to initialize shape
        first_client_id = next(iter(self.received_updates))
        first_weights = self.received_updates[first_client_id]['params']
        
        aggregated_weights = {k: torch.zeros_like(v) for k, v in first_weights.items()}
        
        # 3. Weighted Sum
        for cid, info in self.received_updates.items():
            weight = info['samples'] / total_samples
            client_weights = info['params']
            
            for key in aggregated_weights.keys():
                # Ensure types match (float for aggregation)
                if key in client_weights:
                    aggregated_weights[key] += client_weights[key] * weight

        # 4. Update Global Model
        self.global_model.load_state_dict(aggregated_weights)
        
        # 5. Clear Buffer
        self.received_updates = {}
        
        return aggregated_weights

    def evaluate_global(self, 
                        clean_loader: DataLoader, 
                        backdoor_loader: DataLoader = None) -> Dict[str, float]:
        self.global_model.to(self.device)
        self.global_model.eval()
        
        results = {}

        def run_inference(loader):
            total_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in loader:
                    batch = [t.to(self.device) for t in batch]
                    x = batch[0]
                    y = batch[1]
                    text_lengths = batch[2] if len(batch) > 2 else None

                    output = self.global_model(x, text_lengths=text_lengths)
                    loss = self.criterion(output, y)
                    total_loss += loss.item() * y.size(0)
                    preds = output.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            
            return (total_loss / total) if total else 0.0, (correct / total) if total else 0.0

        if clean_loader:
            loss, acc = run_inference(clean_loader)
            results["clean_loss"] = loss
            results["clean_acc"] = acc
            print(f"  [Server] Clean Accuracy: {acc:.4f}")

        if backdoor_loader:
            _, asr = run_inference(backdoor_loader)
            results["asr"] = asr
            print(f"  [Server] Backdoor ASR:   {asr:.4f}")

        return results
    
    def save_model(self, path: str) -> None:
        torch.save(self.global_model.state_dict(), path)

class FedOptAggregator(FedAvgServer):
    """
    Implements Federated Optimization (Reddi et al., 2020).
    Supports: FedSGD (with Momentum), FedAdam, FedYogi, FedAdagrad.
    """
    def __init__(self, 
                 global_model: torch.nn.Module, 
                 device: str = "cpu",
                 opt_method: str = 'adam', 
                 server_lr: float = 0.01, 
                 betas: tuple = (0.9, 0.99), 
                 tau: float = 1e-3, 
                 momentum: float = 0.9,
                 **kwargs):
        
        super().__init__(global_model, device, **kwargs)
        
        self.opt_method = opt_method.lower()
        self.server_lr = server_lr
        self.betas = betas
        self.tau = tau
        self.momentum = momentum
        
        # Initialize Optimizer State
        # m_t: Momentum (First moment) for Adam/SGD
        # v_t: Variance (Second moment) for Adam/Yogi/Adagrad
        self.m_t = {k: torch.zeros_like(p) for k, p in self.global_model.named_parameters()}
        self.v_t = {k: torch.zeros_like(p) + tau**2 for k, p in self.global_model.named_parameters()}
        
        print(f"Initialized FedOpt ({self.opt_method}) with LR={self.server_lr}, Momentum={self.momentum}")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        1. Aggregates client updates to find the 'Weighted Average'.
        2. Treats (Current - WeightedAvg) as a gradient.
        3. Uses server optimizer to update global model.
        """
        if not self.received_updates:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        self.global_model.to(self.device)
        
        # --- 1. Compute Standard Weighted Average ---
        total_samples = sum(info['samples'] for info in self.received_updates.values())
        first_client_id = next(iter(self.received_updates))
        
        # Initialize accumulator with zeros
        weighted_avg = {
            k: torch.zeros_like(v, device=self.device) 
            for k, v in self.received_updates[first_client_id]['params'].items()
        }
        
        for cid, info in self.received_updates.items():
            weight = info['samples'] / total_samples
            client_params = info['params']
            
            for key in weighted_avg.keys():
                if key in client_params:
                    # Move to GPU and add weighted contribution
                    weighted_avg[key] += client_params[key].to(self.device) * weight

        # --- 2. Compute Pseudo-Gradient ---
        # GRADIENT = CURRENT_WEIGHTS - TARGET_WEIGHTS
        # Subtracting this gradient moves Current towards Target.
        current_params = {k: p for k, p in self.global_model.named_parameters()}
        pseudo_grads = {}
        
        for k, new_w in weighted_avg.items():
            if k in current_params:
                # [CRITICAL FIX] 
                # Direction: Pointing AWAY from the target (so subtraction moves us closer)
                pseudo_grads[k] = current_params[k].data - new_w
        
        # --- 3. Apply Server Optimizer Step ---
        self._server_opt_step(current_params, pseudo_grads)
        
        # --- 4. Cleanup ---
        self.received_updates = {} 
        
        return self.get_params()

    def _server_opt_step(self, params, pseudo_grads):
        beta1, beta2 = self.betas
        
        for k, grad in pseudo_grads.items():
            if k not in self.m_t: continue 
            
            # Ensure states are on the correct device
            self.m_t[k] = self.m_t[k].to(self.device)
            self.v_t[k] = self.v_t[k].to(self.device)
            
            # --- Option A: SGD with Momentum (For 20 Newsgroups Paper) ---
            if self.opt_method == 'sgd':
                # Momentum Buffer Update: v = mu * v + g
                self.m_t[k] = self.momentum * self.m_t[k] + grad
                
                # Step: lr * v
                step = self.server_lr * self.m_t[k]
                
            # --- Option B: Adaptive Optimizers (FedAdam, FedYogi, FedAdagrad) ---
            else:
                # 1. Update Momentum (First Moment)
                self.m_t[k] = beta1 * self.m_t[k] + (1 - beta1) * grad
                
                # 2. Update Variance (Second Moment)
                grad_sq = grad**2
                
                if self.opt_method == 'adam':
                    self.v_t[k] = beta2 * self.v_t[k] + (1 - beta2) * grad_sq
                elif self.opt_method == 'yogi':
                    diff = self.v_t[k] - grad_sq
                    self.v_t[k] = self.v_t[k] - (1 - beta2) * torch.sign(diff) * grad_sq
                elif self.opt_method == 'adagrad':
                    self.v_t[k] = self.v_t[k] + grad_sq

                # 3. Calculate Step
                step = self.server_lr * self.m_t[k] / (torch.sqrt(self.v_t[k]) + self.tau)
            
            # --- Apply Update (In-Place) ---
            # w_new = w_old - step
            params[k].data.sub_(step)