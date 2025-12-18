import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.datasets import fetch_20newsgroups
from transformers import DistilBertTokenizer
import numpy as np
from typing import Dict, List, Optional
from .adapter import DatasetAdapter

class NewsGroupsDataset(DatasetAdapter):
    """
    Adapter for the 20 Newsgroups dataset (20-class classification).
    Uses DistilBERT tokenizer for compatibility with Transformer models.
    """
    def __init__(self, root: str = "./data/newsgroups", max_seq_len: int = 256):
        super().__init__(root, download=True)
        self.max_seq_len = max_seq_len
        self.num_classes = 20
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        self.train_partitions: Dict[int, List[int]] = {}
        # We will use the official test set for global evaluation
        self._test_dataset = None 
        self._train_dataset = None

    def load_datasets(self):
        print("--- Loading 20 Newsgroups ---")
        
        # 1. Fetch Data (Remove headers/footers to prevent overfitting to metadata)
        remove = ('headers', 'footers', 'quotes')
        newsgroups_train = fetch_20newsgroups(subset='train', remove=remove)
        newsgroups_test = fetch_20newsgroups(subset='test', remove=remove)
        
        print(f"Train samples: {len(newsgroups_train.data)} | Test samples: {len(newsgroups_test.data)}")

        # 2. Tokenization Helper
        def tokenize_data(texts, labels):
            encodings = self.tokenizer(
                texts, 
                truncation=True, 
                padding=True, 
                max_length=self.max_seq_len, 
                return_tensors="pt"
            )
            return TensorDataset(encodings['input_ids'], torch.tensor(labels))

        # 3. Create TensorDatasets
        print("Tokenizing Train Set...")
        self._train_dataset = tokenize_data(newsgroups_train.data, newsgroups_train.target)
        
        print("Tokenizing Test Set...")
        self._test_dataset = tokenize_data(newsgroups_test.data, newsgroups_test.target)
        
        # 4. Partitioning (Default to IID)
        # We simulate 100 clients sharing the 11k train docs (~110 docs/client)
        print("Partitioning Train Data (IID)...")
        num_train = len(self._train_dataset)
        indices = np.random.permutation(num_train)
        
        # Hardcoded for 100 clients, but dynamic in get_client_loaders is better
        # Here we just prepare the pool indices
        self.all_train_indices = indices

    def get_client_loaders(self, num_clients: int, batch_size: int, strategy: str = 'iid') -> Dict[int, DataLoader]:
        """
        Splits the training set among `num_clients`.
        """
        loaders = {}
        total_size = len(self._train_dataset)
        
        if strategy == 'iid':
            # Uniform split
            indices = np.random.permutation(total_size)
            split_size = total_size // num_clients
            
            for i in range(num_clients):
                start = i * split_size
                end = start + split_size if i < num_clients - 1 else total_size
                client_indices = indices[start:end]
                
                subset = Subset(self._train_dataset, client_indices)
                loaders[i] = DataLoader(subset, batch_size=batch_size, shuffle=True)
                
                # Store partition for debugging
                self.train_partitions[i] = client_indices.tolist()
                
        else:
            raise NotImplementedError("Only IID supported for Newsgroups currently.")

        return loaders

    def get_test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self._test_dataset, batch_size=batch_size)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size