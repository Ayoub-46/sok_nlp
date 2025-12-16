import torch
import copy
from typing import Dict, Any
from .client import BenignClient

class FedProxClient(BenignClient):
    """
    FedProx Client: Adds a proximal term to the loss to handle non-IID data.
    Ref: Li et al., "Federated Optimization in Heterogeneous Networks"
    """
    def __init__(self, mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Trains locally with Proximal Regularization.
        """
        self.model.to(self.device)
        self.model.train()
        
        # 1. Capture Global State (The Anchor)
        # We need a deep copy of the weights at the start of the round
        global_params = copy.deepcopy([p.detach() for p in self.model.parameters()])
        
        optimizer = self.optimizer_cls(self.model.parameters())
        
        epoch_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            running_loss = 0.0
            
            for batch in self.train_loader:
                batch = [t.to(self.device) for t in batch]
                x, y = batch[0], batch[1]
                text_lengths = batch[2] if len(batch) > 2 else None
                
                optimizer.zero_grad()
                
                # Forward Pass
                output = self.model(x, text_lengths=text_lengths)
                task_loss = self.criterion(output, y)
                
                # 2. Calculate Proximal Term
                # prox_loss = sum(||w - w_t||^2)
                prox_loss = 0.0
                for param, global_param in zip(self.model.parameters(), global_params):
                    prox_loss += torch.norm(param - global_param) ** 2
                
                # Total Loss
                loss = task_loss + (self.mu / 2) * prox_loss
                
                loss.backward()
                
                # Clip gradients (still important for LSTM)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Logging (We track the task loss, usually)
                running_loss += loss.item() * y.size(0)
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            epoch_loss = running_loss / total

        return {
            "client_id": self.id,
            "train_loss": epoch_loss,
            "train_acc": correct / total if total > 0 else 0.0,
            "samples": total
        }