import torch
import copy
from typing import Dict, Any
from torch.utils.data import DataLoader

from ..fl.client import BenignClient
from ..datasets.backdoor import BackdoorNLPDataset
from ..attacks.triggers import Trigger

class ModelReplacementClient(BenignClient):
    """
    Implements the Model Replacement (Constrain-and-Scale) attack for NLP.
    
    Logic: W_submit = W_global + Scale * (W_malicious - W_global)
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
        
        # Attack Parameters
        self.target_label = attack_config.get('target_label', 0)
        self.poison_fraction = attack_config.get('poison_fraction', 0.5)
        self.scaling_factor = attack_config.get('scaling_factor', 10.0)
        
        # Timing
        self.malicious_epochs = attack_config.get('malicious_epochs', 5) # Train harder than benign clients
        self.attack_start_round = attack_config.get('attack_start_round', 1)
        self.attack_end_round = attack_config.get('attack_end_round', float('inf'))

    def _create_poisoned_loader(self) -> DataLoader:
        """
        Wraps the benign dataset with the Backdoor logic using our Trigger object.
        """
        clean_dataset = self.train_loader.dataset
        
        poisoned_dataset = BackdoorNLPDataset(
            original_dataset=clean_dataset,
            trigger_fn=self.trigger.apply, # Delegate to the Trigger object
            target_label=self.target_label,
            poison_fraction=self.poison_fraction,
            poison_exclude_target=True
        )

        return DataLoader(
            poisoned_dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True
        )

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Executes the Model Replacement attack.
        """
        # 1. Check Attack Window
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            # Behave like a benign client if outside the attack window
            return super().local_train(epochs, round_idx)
        
        print(f"\n  [Attack] Model Replacement Client {self.id} active in Round {round_idx}!")

        # 2. Capture Global Model State (W_global)
        # Keep on CPU to save VRAM, or clone on device if small enough.
        global_model_params = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        
        # 3. Swap Data Loader (Benign -> Poisoned)
        original_loader = self.train_loader
        self.train_loader = self._create_poisoned_loader()
        
        # 4. Train to obtain W_malicious
        # We reuse the super() loop because it correctly handles NLP inputs (x, y, lengths/masks)
        # We ignore the 'epochs' arg passed by Runner and use 'malicious_epochs'
        metrics = super().local_train(epochs=self.malicious_epochs, round_idx=round_idx)
        
        # Restore benign loader
        self.train_loader = original_loader
        
        # 5. Apply Scaling: W_submit = W_global + Scale * (W_mal - W_global)
        print(f"  [Attack] Scaling updates by factor: {self.scaling_factor}")
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_model_params:
                    # Move global param to same device for calculation
                    global_val = global_model_params[name].to(self.device)
                    malicious_val = param.data
                    
                    # Calculate vector
                    update_vector = malicious_val - global_val
                    
                    # Scale and Apply
                    scaled_update = update_vector * self.scaling_factor
                    param.data.copy_(global_val + scaled_update)
        
        # Add flag for the runner to log
        metrics['is_attacker'] = True
        return metrics