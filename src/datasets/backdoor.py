from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, Any
import numpy as np
import torch 

class BackdoorNLPDataset(Dataset):
    """
    A unified Backdoor Wrapper for NLP Tasks.
    
    Key Features:
    1. NLP-Native: Handles variable tuple lengths (e.g., (x, y), (x, y, len), (x, y, mask)).
    2. Memory Efficient: Applies triggers on-the-fly (no massive pre-caching).
    3. Type Agnostic: Works with Tensors (Sentiment140) or Strings (Shakespeare raw), 
       provided the trigger_fn matches the input type.
    """
    def __init__(self,
                 original_dataset: Dataset,
                 trigger_fn: Callable[[Any], Any],
                 target_label: int,
                 poison_fraction: float = 1.0,
                 seed: int = 42,
                 poison_exclude_target: bool = True): 
        
        self.original_dataset = original_dataset
        self.trigger_fn = trigger_fn
        self.target_label = target_label
        self.poison_fraction = poison_fraction
        
        # --- 1. Determine Eligible Indices ---
        dataset_size = len(self.original_dataset)
        all_indices = np.arange(dataset_size)
        eligible_indices = all_indices

        if poison_exclude_target:
            # fast-path: try to access .targets (standard in Torch/Numpy datasets)
            if hasattr(self.original_dataset, "targets"):
                targets = np.array(self.original_dataset.targets)
                # Ensure targets are on CPU/Numpy for boolean indexing
                if isinstance(targets, torch.Tensor):
                    targets = targets.cpu().numpy()
                eligible_indices = all_indices[targets != self.target_label]
            else:
                # Warning: Without .targets, we assume all are eligible to avoid 
                # iterating the whole dataset (slow for NLP).
                print("Warning: .targets not found. Poisoning checks may overwrite existing target labels.")

        # --- 2. Select Poisoned Indices ---
        num_poisoned = int(len(eligible_indices) * self.poison_fraction)
        rng = np.random.RandomState(seed)
        
        # We use a Set for O(1) lookup in __getitem__
        self.poisoned_indices = set(rng.choice(eligible_indices, num_poisoned, replace=False))
        
        print(f"Backdoor Injection Ready: {len(self.poisoned_indices)} samples will be poisoned ({self.poison_fraction*100:.1f}%).")

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        # Get raw sample from source
        sample = self.original_dataset[index]
        
        # --- 3. Flexible Unpacking ---
        # NLP datasets often return (input_ids, label, attention_mask, ...)
        # We assume index 0 is Input, index 1 is Label.
        x = sample[0]
        y = sample[1]
        extras = sample[2:] # Capture any extra metadata (lengths, masks)
        
        # --- 4. Apply Backdoor ---
        if index in self.poisoned_indices:
            # Apply Trigger (Function must handle the specific type of x: Tensor or Str)
            x = self.trigger_fn(x)
            
            # Apply Target Label
            # Handle Scalar vs Tensor label
            if isinstance(y, torch.Tensor):
                y = torch.tensor(self.target_label, dtype=y.dtype, device=y.device)
            else:
                y = self.target_label

        # Re-pack and return
        return (x, y) + extras