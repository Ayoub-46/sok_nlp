from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import copy 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaseClient(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def num_samples(self) -> int:
        pass

    @abstractmethod
    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def local_evaluate(self) -> Dict[str, Any]:
        pass

class BenignClient(BaseClient):
    """
    Standard Federated Learning Client.
    Performs local training using a clean (benign) dataset.
    """
    def __init__(self, 
                 client_id: int, 
                 model: nn.Module, 
                 train_loader: DataLoader, 
                 test_loader: DataLoader = None, 
                 device: str = "cpu",
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 lr: float = 0.1,
                 optimizer_cls=torch.optim.SGD):
        
        self.id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.lr = lr
        self.optimizer_cls = optimizer_cls

    def get_id(self) -> int:
        return self.id

    def num_samples(self) -> int:
        return len(self.train_loader.dataset)

    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Overwrites local model weights with the global model weights.
        """
        self.model.load_state_dict(state_dict)

    def get_params(self) -> Dict[str, torch.Tensor]:
        """
        Returns a deep copy of the model weights on CPU.
        """
        self.model.cpu()
        return copy.deepcopy(self.model.state_dict())

    def local_train(self, epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        """
        Trains the model locally on self.train_loader.
        """
        self.model.to(self.device)
        self.model.train()
        
        optimizer = self.optimizer_cls(self.model.parameters())
        epoch_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            running_loss = 0.0
            
            for batch in self.train_loader:
                # --- Dynamic Unpacking for NLP Tasks ---
                # Batch can be (x, y) [Shakespeare] or (x, y, len) [Sentiment]
                batch = [t.to(self.device) for t in batch]
                
                x = batch[0]
                y = batch[1]
                # If lengths are present (index 2), pass them; else None
                text_lengths = batch[2] if len(batch) > 2 else None
                
                optimizer.zero_grad()
                
                # Forward pass (model handles None lengths gracefully)
                output = self.model(x, text_lengths=text_lengths)
                
                loss = self.criterion(output, y)
                loss.backward()
                
                # Gradient Clipping (Crucial for LSTMs to prevent exploding gradients)
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Reduce clip to 1.0
                
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"  [!] Client {self.id}: Gradient Explosion detected! Skipping step.")
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
                
                running_loss += loss.item() * y.size(0)
                
                # Metrics
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            epoch_loss = running_loss / total

        # Return local metrics
        return {
            "client_id": self.id,
            "train_loss": epoch_loss,
            "train_acc": correct / total if total > 0 else 0.0,
            "samples": total
        }

    def local_evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the current model on the local test set (if available).
        """
        if self.test_loader is None:
            return {}

        self.model.to(self.device)
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = [t.to(self.device) for t in batch]
                x = batch[0]
                y = batch[1]
                text_lengths = batch[2] if len(batch) > 2 else None
                
                output = self.model(x, text_lengths=text_lengths)
                loss = self.criterion(output, y)
                
                total_loss += loss.item() * y.size(0)
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return {
            "client_id": self.id,
            "eval_loss": total_loss / total if total > 0 else 0.0,
            "eval_acc": correct / total if total > 0 else 0.0
        }
