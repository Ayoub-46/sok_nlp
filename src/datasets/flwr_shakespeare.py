import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset, Dataset
from .adapter import DatasetAdapter

class TokenizedShakespeare(Dataset):
    """
    Internal Wrapper:
    1. Converts HF Dictionary format {'x':..., 'y':...} -> Tuple (x, y).
    2. Tokenizes strings into Integers.
    3. Exposes .targets for the Backdoor optimization.
    """
    def __init__(self, hf_dataset, char_to_int, seq_len=80):
        self.data = hf_dataset
        self.char_to_int = char_to_int
        self.default_idx = self.char_to_int.get(' ', 1)
        
        # Pre-cache targets (Label Encoding)
        # We iterate once so BackdoorDataset doesn't have to scan everything
        print("  -> Pre-caching targets...")
        self.targets = [self.char_to_int.get(c, self.default_idx) for c in hf_dataset['y']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize Input (String -> List[Int])
        idx_x = [self.char_to_int.get(c, self.default_idx) for c in item['x']]
        
        # Get Target (already cached as Int)
        label = self.targets[idx]
        
        # Return TUPLE of Tensors
        # This matches the Sentiment140 format and satisfies BackdoorNLPDataset
        return torch.tensor(idx_x, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class FlwrShakespeareDataset(DatasetAdapter):
    """
    Adapter for flwrlabs/shakespeare.
    Compatible with BackdoorNLPDataset without custom collate_fn.
    """
    def __init__(self, root: str = "data/flwr_shakespeare", download: bool = True):
        super().__init__(root, download, None, None)
        
        # Standard LEAF Vocabulary
        self.leaf_chars = "\n !\"&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        self.pad_token = "<PAD>"
        self.vocab = [self.pad_token] + list(self.leaf_chars)
        self.char_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        self.train_partitions = {} 
        self.test_partitions = {}
        self.tokenized_dataset = None 

    def load_datasets(self) -> None:
        print("--- Loading flwrlabs/shakespeare ---")
        hf_data = load_dataset("flwrlabs/shakespeare")['train']

        # [CRITICAL] Wrap immediately to Standardize Output as Tensors
        self.tokenized_dataset = TokenizedShakespeare(hf_data, self.char_to_int)

        print("Partitioning data by character_id...")
        all_user_indices = self._partition_by_char(hf_data)
        
        # Standard LEAF Split (80/20)
        for user, indices in all_user_indices.items():
            n_samples = len(indices)
            if n_samples < 2: continue
            
            split_idx = int(n_samples * 0.8)
            if split_idx == 0 and n_samples > 0: split_idx = 1
                
            self.train_partitions[user] = indices[:split_idx]
            self.test_partitions[user] = indices[split_idx:]
            
        # Centralized Test Set
        all_test_indices = []
        for user in self.test_partitions:
            all_test_indices.extend(self.test_partitions[user])
            
        self._test_dataset = Subset(self.tokenized_dataset, all_test_indices)
        self._train_dataset = self.tokenized_dataset
        
        print(f"Loaded {len(self.train_partitions)} clients. Test set size: {len(self._test_dataset)}")

    def _partition_by_char(self, hf_split):
        try:
            df = hf_split.select_columns(["character_id"]).to_pandas()
            return df.groupby("character_id").indices.to_dict()
        except Exception as e:
            print(f"Pandas grouping failed: {e}. Falling back to slow loop.")
            indices = {}
            for i, item in enumerate(hf_split):
                char = item['character_id']
                if char not in indices: indices[char] = []
                indices[char].append(i)
            return indices

    def get_client_loaders(self, num_clients: int, batch_size: int = 64, seed: int = 0, **kwargs) -> dict:
        self.setup()
        all_users = sorted(list(self.train_partitions.keys()))
        
        rng = np.random.RandomState(seed)
        if num_clients < len(all_users):
            selected_users = rng.choice(all_users, num_clients, replace=False)
        else:
            selected_users = all_users
            
        loaders = {}
        for i, user_id in enumerate(selected_users):
            indices = self.train_partitions[user_id]
            if len(indices) == 0: continue
            
            subset = Subset(self.tokenized_dataset, indices)
            loaders[i] = DataLoader(subset, batch_size=batch_size, shuffle=True)
            
        return loaders
    
    def get_vocab_size(self):
        return self.vocab_size