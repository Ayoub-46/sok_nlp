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
    
    Responsibilities:
    1. Buffer updates from clients.
    2. Aggregate weights using weighted averaging.
    3. Evaluate Global Model (Clean Accuracy + Backdoor ASR).
    """
    def __init__(self, 
                 global_model: nn.Module, 
                 device: str = "cpu",
                 criterion: nn.Module = nn.CrossEntropyLoss()):
        
        self.global_model = global_model
        self.device = device
        self.criterion = criterion
        
        # Buffer to store client updates: List of (state_dict, num_samples)
        self.updates_buffer: List[Tuple[Dict[str, torch.Tensor], int]] = []

    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.global_model.load_state_dict(state_dict)

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Returns CPU-based state dict to send to clients."""
        self.global_model.cpu()
        return copy.deepcopy(self.global_model.state_dict())

    def receive_update(self, state_dict: Dict[str, torch.Tensor], num_samples: int) -> None:
        """
        Collects a client's update. Call this during the round.
        """
        # Store deepcopy to avoid reference issues
        self.updates_buffer.append((copy.deepcopy(state_dict), num_samples))

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Performs FedAvg: w_global = sum(w_k * n_k) / sum(n_k)
        Resets the buffer afterwards.
        """
        if not self.updates_buffer:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        # 1. Calculate total samples
        total_samples = sum(n for _, n in self.updates_buffer)
        
        # 2. Initialize aggregated weights with the first update (weighted)
        first_weights, first_n = self.updates_buffer[0]
        aggregated_weights = copy.deepcopy(first_weights)
        
        for key in aggregated_weights.keys():
            # Apply weight for first client
            aggregated_weights[key] = aggregated_weights[key] * (first_n / total_samples)
            
            # Add remaining clients
            for i in range(1, len(self.updates_buffer)):
                other_weights, other_n = self.updates_buffer[i]
                # Ensure correct type (float/long) handling implicitly
                weighted_param = other_weights[key] * (other_n / total_samples)
                aggregated_weights[key] += weighted_param

        # 3. Update Global Model
        self.global_model.load_state_dict(aggregated_weights)
        
        # 4. Clear Buffer
        self.updates_buffer = []
        
        return aggregated_weights

    def evaluate_global(self, 
                        clean_loader: DataLoader, 
                        backdoor_loader: DataLoader = None) -> Dict[str, float]:
        """
        Centralized Evaluation.
        Returns: {'clean_acc': ..., 'clean_loss': ..., 'asr': ...}
        """
        self.global_model.to(self.device)
        self.global_model.eval()
        
        results = {}

        # --- Helper for Evaluation Loop ---
        def run_inference(loader):
            total_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in loader:
                    # Dynamic Unpacking (Matches Client Logic)
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

        # 1. Clean Evaluation (Main Task Utility)
        if clean_loader:
            loss, acc = run_inference(clean_loader)
            results["clean_loss"] = loss
            results["clean_acc"] = acc
            print(f"  [Server] Clean Accuracy: {acc:.4f}")

        # 2. Backdoor Evaluation (Attack Success Rate)
        if backdoor_loader:
            # Note: backdoor_loader samples already have TARGET labels.
            # So 'accuracy' on this dataset IS the Attack Success Rate (ASR).
            _, asr = run_inference(backdoor_loader)
            results["asr"] = asr
            print(f"  [Server] Backdoor ASR:   {asr:.4f}")

        return results
    
    def save_model(self, path: str) -> None:
        torch.save(self.global_model.state_dict(), path)

class FedOptAggregator(FedAvgServer):
    """
    Implements Federated Optimization (FedAdam, FedYogi, FedAdagrad).
    Computes a 'pseudo-gradient' (averaged update) and applies it using a server-side optimizer.
    """
    def __init__(self, 
                 global_model: torch.nn.Module, 
                 device: str = "cpu",
                 opt_method: str = 'adam', 
                 server_lr: float = 0.01, 
                 betas: tuple = (0.9, 0.99), 
                 tau: float = 1e-3, 
                 **kwargs):
        
        # Initialize parent (FedAvgServer)
        super().__init__(global_model, device, **kwargs)
        
        self.opt_method = opt_method.lower()
        self.server_lr = server_lr
        self.betas = betas
        self.tau = tau
        
        # Initialize Server-Side Optimizer State (Momentum & Velocity)
        # We ensure these states are on the same device as the model
        self.m_t = {k: torch.zeros_like(p) for k, p in self.global_model.named_parameters()}
        self.v_t = {k: torch.zeros_like(p) + tau**2 for k, p in self.global_model.named_parameters()}
        
    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Overrides standard FedAvg aggregation.
        """
        if not self.updates_buffer:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        # [FIX] Ensure the global model is on the correct device (GPU)
        # It was moved to CPU by get_params() earlier in the round.
        self.global_model.to(self.device)
        
        # --- 1. Compute Standard Weighted Average (Intermediate Step) ---
        total_samples = sum(n for _, n in self.updates_buffer)
        
        weighted_avg = {}
        first_weights, _ = self.updates_buffer[0]
        
        for key in first_weights.keys():
            # Initialize accumulator on GPU
            acc = torch.zeros_like(first_weights[key], device=self.device)
            
            for client_weights, n_k in self.updates_buffer:
                weight = n_k / total_samples
                # Move client tensor to GPU for math
                w_k = client_weights[key].to(self.device)
                acc += w_k * weight
            
            weighted_avg[key] = acc

        # --- 2. Compute Pseudo-Gradient (Delta) ---
        # Now self.global_model is on GPU, so current_params will be on GPU
        current_params = {k: p for k, p in self.global_model.named_parameters()}
        pseudo_grads = {}
        
        for k, new_w in weighted_avg.items():
            if k in current_params:
                # [SUCCESS] Both new_w and current_params[k] are now on cuda:0
                pseudo_grads[k] = new_w - current_params[k].data
        
        # --- 3. Apply Server Optimizer Step ---
        self._server_opt_step(current_params, pseudo_grads)
        
        # --- 4. Cleanup ---
        self.updates_buffer = [] 
        
        return self.get_params()

    def _server_opt_step(self, params, pseudo_grads):
        beta1, beta2 = self.betas
        
        for k, grad in pseudo_grads.items():
            if k not in self.m_t: continue 
            
            # Ensure state is on correct device
            self.m_t[k] = self.m_t[k].to(self.device)
            self.v_t[k] = self.v_t[k].to(self.device)
            
            # --- Momentum (m_t) ---
            # m_t = beta1 * m_t + (1 - beta1) * grad
            self.m_t[k] = beta1 * self.m_t[k] + (1 - beta1) * grad
            
            # --- Velocity (v_t) ---
            grad_sq = grad**2
            
            if self.opt_method == 'adam':
                self.v_t[k] = beta2 * self.v_t[k] + (1 - beta2) * grad_sq
            
            elif self.opt_method == 'yogi':
                diff = self.v_t[k] - grad_sq
                self.v_t[k] = self.v_t[k] - (1 - beta2) * torch.sign(diff) * grad_sq
            
            elif self.opt_method == 'adagrad':
                self.v_t[k] = self.v_t[k] + grad_sq

            # --- Update Weights ---
            # w_{t+1} = w_t + lr * m_t / (sqrt(v_t) + tau)
            # Note: We ADD because 'grad' here is actually the Update Delta (Direction), 
            # not the negative gradient of the loss surface.
            step = self.server_lr * self.m_t[k] / (torch.sqrt(self.v_t[k]) + self.tau)
            params[k].data.add_(step)