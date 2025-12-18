import torch
import copy
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from ..fl.client import BenignClient
from ..datasets.backdoor import BackdoorNLPDataset
from ..attacks.triggers import Trigger

class NeurotoxinClient(BenignClient):
    """
    Implements the Neurotoxin attack for NLP.
    
    Constraint-based attack that identifies "important" parameters (those that change 
    the most in the global model) and prevents the attacker from modifying them.
    This preserves main-task accuracy while injecting the backdoor into "unused" capacity.
    """
    def __init__(self, 
                 client_id: int, 
                 model: torch.nn.Module, 
                 train_loader: DataLoader, 
                 device: str,
                 trigger: Trigger,
                 attack_config: Dict,
                 **kwargs):
        
        super().__init__(client_id, model, train_loader, device=device, **kwargs)
        
        self.trigger = trigger
        self.attack_config = attack_config
        
        # Attack Params
        self.target_label = attack_config.get('target_label', 0)
        self.poison_fraction = attack_config.get('poison_fraction', 0.25)
        self.mask_k_percent = attack_config.get('mask_k_percent', 0.05) # Top 5% params are protected
        self.malicious_epochs = attack_config.get('malicious_epochs', 10)
        
        self.attack_start_round = attack_config.get('attack_start_round', 1)
        self.attack_end_round = attack_config.get('attack_end_round', float('inf'))

    def _create_poisoned_loader(self) -> DataLoader:
        clean_dataset = self.train_loader.dataset
        
        poisoned_dataset = BackdoorNLPDataset(
            original_dataset=clean_dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_label,
            poison_fraction=self.poison_fraction,
            poison_exclude_target=True
        )

        return DataLoader(
            poisoned_dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True
        )

    def compute_grad_mask(self, prev_global_grad: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Computes a binary mask where 1 = trainable (unimportant), 0 = frozen (important).
        Importance is derived from the magnitude of the previous global update relative to weight magnitude.
        """
        if prev_global_grad is None:
            return None

        importances = []
        key_to_delta = {}
        eps = 1e-12

        # 1. Calculate Importance Scores
        for name, param in self.model.named_parameters():
            if name not in prev_global_grad:
                continue
            
            # [FIX] Ensure everything is on the same device.
            # param.data is already on self.device because we moved the model in local_train BEFORE calling this.
            d_tensor = prev_global_grad[name].to(self.device) 
            p_tensor = param.data 
            
            # Metric: |Delta| / (|Weight| + epsilon)
            importance = (d_tensor.abs() / (p_tensor.abs() + eps)).flatten()
            importances.append(importance)
            key_to_delta[name] = d_tensor

        if not importances:
            return None

        # 2. Determine Threshold for Top-K%
        all_importances = torch.cat(importances)
        k = max(1, int(self.mask_k_percent * all_importances.numel()))
        
        # Get the threshold value that separates the top k elements
        threshold = torch.topk(all_importances, k, largest=True, sorted=True)[0][-1]

        # 3. Create Mask
        grad_mask = {}
        for name, d_tensor in key_to_delta.items():
            param = dict(self.model.named_parameters())[name]
            importance = (d_tensor.abs() / (param.data.abs() + eps))
            
            # Mask = 1 (Allow training), Mask = 0 (Block training)
            mask = (importance < threshold).float()
            grad_mask[name] = mask
            
        return grad_mask

    def local_evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the current malicious model on a 100% poisoned version of the 
        local training data to check Attack Success Rate (ASR).
        """
        self.model.to(self.device)
        self.model.eval()
        
        base_loader = self.test_loader if self.test_loader else self.train_loader
        clean_dataset = base_loader.dataset

        eval_poisoned_dataset = BackdoorNLPDataset(
            original_dataset=clean_dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_label,
            poison_fraction=1.0, 
            poison_exclude_target=True
        )
        
        eval_loader = DataLoader(eval_poisoned_dataset, batch_size=32, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = [t.to(self.device) for t in batch]
                x = batch[0]
                y = batch[1]
                text_lengths = batch[2] if len(batch) > 2 else None
                
                output = self.model(x, text_lengths=text_lengths)
                preds = output.argmax(dim=1)
                
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        asr = correct / total if total > 0 else 0.0
        print(f"  [Client {self.id} Debug] Local ASR (Success Rate): {asr:.4f}")
        return {"local_asr": asr}
    
    def local_train(self, epochs: int, round_idx: int, prev_global_grad: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        
        # 1. Check Window
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            return super().local_train(epochs, round_idx)

        print(f"\n  [Neurotoxin] Client {self.id} attacking Round {round_idx}")

        # [FIX] Move Model to GPU FIRST. 
        # This ensures param.data is on the correct device when compute_grad_mask accesses it.
        self.model.to(self.device)
        self.model.train()

        # 2. Compute Mask
        grad_mask = self.compute_grad_mask(prev_global_grad)
        
        if grad_mask is None:
            print("  [Neurotoxin] No previous global grad available. Attacking without mask.")
        else:
            masked_count = sum([ (1-m).sum().item() for m in grad_mask.values() ])
            total_params = sum([ m.numel() for m in grad_mask.values() ])
            print(f"  [Neurotoxin] Mask generated. {masked_count}/{total_params} parameters frozen (Top {self.mask_k_percent*100}%).")

        # 3. Setup Poisoning
        original_loader = self.train_loader
        self.train_loader = self._create_poisoned_loader()
        
        # 4. Custom Training Loop (To inject Mask)
        optimizer = self.optimizer_cls(self.model.parameters())
        
        epoch_loss = 0.0
        total = 0

        for _ in range(self.malicious_epochs):
            running_loss = 0.0
            
            for batch in self.train_loader:
                batch = [t.to(self.device) for t in batch]
                x, y = batch[0], batch[1]
                text_lengths = batch[2] if len(batch) > 2 else None
                
                optimizer.zero_grad()
                output = self.model(x, text_lengths=text_lengths)
                loss = self.criterion(output, y)
                loss.backward()
                
                # --- APPLY MASK ---
                if grad_mask is not None:
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.grad is not None and name in grad_mask:
                                # Multiply grad by 0 if masked, 1 if allowed
                                param.grad.mul_(grad_mask[name])
                # ------------------

                # Clip grads (important for NLP stability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item() * y.size(0)
                total += y.size(0)
            
            epoch_loss = running_loss / total if total > 0 else 0.0

        # Restore
        self.train_loader = original_loader

        # Debug Evaluation
        self.local_evaluate()

        return {
            "client_id": self.id,
            "train_loss": epoch_loss,
            "samples": total,
            "is_attacker": True
        }