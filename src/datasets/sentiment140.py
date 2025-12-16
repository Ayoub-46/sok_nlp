import torch
import pandas as pd
import numpy as np
import re
import string
import os
from collections import Counter
from typing import Dict, List, Optional
from torch.utils.data import TensorDataset, DataLoader, Subset

# Try importing transformers, but don't crash if not installed (unless used)
try:
    from transformers import DistilBertTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .adapter import DatasetAdapter

class Sentiment140Dataset(DatasetAdapter):
    """
    Adapter for Sentiment140 dataset.
    Supports two modes:
    1. 'LSTM Mode' (Default): Builds custom vocab, cleans text manually, loads GloVe.
    2. 'BERT Mode' (via tokenizer_name): Uses HF Tokenizer, skips cleaning (BERT handles it).
    """
    def __init__(self, root: str = "data/sentiment140", download: bool = True, tokenizer_name: str = None):
        super().__init__(root, download, None, None)
        
        # Configuration
        self.tokenizer_name = tokenizer_name
        self.hf_tokenizer = None
        
        self.max_vocab_size = 50000  
        self.max_seq_len = 100 # Max length for both LSTM and BERT
        self.embedding_dim = 100
        self.min_samples_per_user = 15 # Filter out users with too few tweets
        
        # LSTM specific state
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.embedding_weights = None
        self.pad_idx = 0
        self.unk_idx = 1
        
        # BERT specific state
        if self.tokenizer_name:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers library required for tokenizer_name usage. Run `pip install transformers`.")
            print(f"--- Loading HF Tokenizer: {self.tokenizer_name} ---")
            self.hf_tokenizer = DistilBertTokenizer.from_pretrained(self.tokenizer_name)
            self.pad_idx = self.hf_tokenizer.pad_token_id
            self.vocab_size = self.hf_tokenizer.vocab_size
        
        # Natural Partition Indices: { client_id: [global_idx1, global_idx2...] }
        self.natural_train_partitions: Dict[int, List[int]] = {}
        self._is_loaded = False
        
        # Path
        self.csv_path = os.path.join(self.root, 'training.1600000.processed.noemoticon.csv')

    def load_datasets(self) -> None:
        if self._is_loaded: return
        print("--- Loading Sentiment140 Data ---")

        if not os.path.exists(self.csv_path):
             raise FileNotFoundError(f"Sentiment140 file not found at {self.csv_path}. Please download it.")

        cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
        print("Reading CSV (this may take a moment)...")
        # use latin-1 because the dataset contains legacy encoding
        df = pd.read_csv(self.csv_path, encoding='latin-1', names=cols)
        
        # Map target: 0=Negative, 4=Positive -> 0=Negative, 1=Positive
        df['target'] = df['target'].replace(4, 1)

        print("Performing GLOBAL SHUFFLE...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Cleaning Logic (Only for LSTM)
        # For BERT, we usually keep raw text (including @mentions) or do minimal cleaning
        if self.hf_tokenizer:
            # Minimal cleaning for BERT (just empty strings check)
            df['clean_text'] = df['text'].astype(str)
        else:
            print("Cleaning text (Regex)...")
            df['clean_text'] = df['text'].apply(self._clean_text)

        # Drop empty
        initial_len = len(df)
        df = df[df['clean_text'].str.strip().astype(bool)]
        print(f"Dropped {initial_len - len(df)} empty tweets.")

        # --- Natural User Partitioning ---
        print("Grouping by user (Natural Partitioning)...")
        
        user_counts = df['user'].value_counts()
        valid_users = user_counts[user_counts >= self.min_samples_per_user].index.tolist()
        df = df[df['user'].isin(valid_users)]
        
        train_texts, train_labels = [], []
        test_texts, test_labels = [], []
        
        current_train_idx = 0
        client_id_counter = 0
        
        grouped = df.groupby('user')
        
        for user, group in grouped:
            group = group.sample(frac=1, random_state=42).reset_index(drop=True)
            
            user_texts = group['clean_text'].values
            user_targets = group['target'].values
            
            # 80/20 Split
            n_samples = len(user_texts)
            n_train = int(0.8 * n_samples)
            
            u_train_txt = user_texts[:n_train]
            u_train_y = user_targets[:n_train]
            u_test_txt = user_texts[n_train:]
            u_test_y = user_targets[n_train:]
            
            # Store indices for Client selection
            indices = list(range(current_train_idx, current_train_idx + len(u_train_txt)))
            self.natural_train_partitions[client_id_counter] = indices
            
            train_texts.extend(u_train_txt)
            train_labels.extend(u_train_y)
            test_texts.extend(u_test_txt)
            test_labels.extend(u_test_y)
            
            current_train_idx += len(u_train_txt)
            client_id_counter += 1

        print(f"Processed {client_id_counter} natural clients from {len(df)} tweets.")

        # --- Vocabulary & Embeddings ---
        if self.hf_tokenizer:
            print("Using HF Tokenizer (Skipping Vocab Build & GloVe Load)")
        else:
            self._build_vocab(train_texts)
            self._load_glove_embeddings(dim=self.embedding_dim)

        # --- Tensor Processing ---
        print("Tokenizing and creating tensors...")
        # Note: This method now handles both LSTM (custom) and BERT (hf) logic
        x_train, y_train, extra_train = self._process_text_to_tensor(train_texts, train_labels)
        x_test, y_test, extra_test = self._process_text_to_tensor(test_texts, test_labels)

        # --- Create Datasets ---
        # extra_train is 'lengths' (LSTM) or 'attention_mask' (BERT)
        self._train_dataset = TensorDataset(x_train, y_train, extra_train)
        self._train_dataset.targets = y_train.numpy()
        
        self._test_dataset = TensorDataset(x_test, y_test, extra_test)
        self._test_dataset.targets = y_test.numpy()

        self._is_loaded = True
        print("Sentiment140 preparation complete.")

    def get_client_loaders(self, num_clients: int, batch_size: int = 32, strategy: str = "iid", seed: int = 0, **strategy_args) -> Dict[int, DataLoader]:
        self.setup()
        
        if strategy == "natural":
            print(f"--- Generating {num_clients} Client Loaders using Natural User Partition ---")
            loaders = {}
            available_clients = list(self.natural_train_partitions.keys())
            
            if num_clients > len(available_clients):
                num_clients = len(available_clients)
            
            rng = np.random.RandomState(seed)
            selected_user_ids = rng.choice(available_clients, num_clients, replace=False)

            for i, original_user_id in enumerate(selected_user_ids):
                indices = self.natural_train_partitions[original_user_id]
                subset = Subset(self.train_dataset, indices)
                loaders[i] = DataLoader(subset, batch_size=batch_size, shuffle=True)
            return loaders
        else:
            return super().get_client_loaders(num_clients, batch_size, strategy, seed, **strategy_args)

    # --- Internals ---

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()

    def _build_vocab(self, texts):
        print(f"Building Vocabulary (Max: {self.max_vocab_size})...")
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        
        most_common = counter.most_common(self.max_vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
        print(f"Vocabulary size: {len(self.word2idx)}")

    def _process_text_to_tensor(self, texts, labels):
        if self.hf_tokenizer:
            # --- BERT Processing ---
            # HF Tokenizer handles truncation/padding to max_length
            encodings = self.hf_tokenizer(
                list(texts),
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encodings['input_ids']
            attention_masks = encodings['attention_mask'] # This is the "extra" tensor
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            return input_ids, labels_tensor, attention_masks
            
        else:
            # --- LSTM Processing ---
            x_list = []
            lengths = []
            
            for text in texts:
                words = text.split()
                indices = [self.word2idx.get(w, self.unk_idx) for w in words]
                
                # Minimum length 1 safety check
                if len(indices) == 0:
                    indices = [self.pad_idx]

                actual_len = min(len(indices), self.max_seq_len)
                lengths.append(actual_len)
                
                # Pad/Truncate
                if len(indices) < self.max_seq_len:
                    indices += [self.pad_idx] * (self.max_seq_len - len(indices))
                else:
                    indices = indices[:self.max_seq_len]
                x_list.append(indices)
                
            return (torch.tensor(x_list, dtype=torch.long), 
                    torch.tensor(labels, dtype=torch.long), 
                    torch.tensor(lengths, dtype=torch.long))

    def _load_glove_embeddings(self, dim=100):
        glove_path = os.path.join(os.path.dirname(self.root), 'glove', f'glove.6B.{dim}d.txt')
        if not os.path.exists(glove_path):
            print(f"Warning: GloVe not found at {glove_path}. Initializing random.")
            self.embedding_weights = torch.randn(len(self.word2idx), dim)
            return

        print(f"Loading GloVe embeddings from {glove_path}...")
        embeddings_index = {}
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except ValueError: continue
        
        # Init random
        matrix = np.random.normal(scale=0.6, size=(len(self.word2idx), dim))
        # Zero out PAD
        matrix[self.pad_idx] = np.zeros(dim)
        
        hits = 0
        for word, idx in self.word2idx.items():
            vec = embeddings_index.get(word)
            if vec is not None:
                matrix[idx] = vec
                hits += 1
                
        self.embedding_weights = torch.tensor(matrix, dtype=torch.float32)
        print(f"GloVe loaded. Coverage: {hits}/{len(self.word2idx)}")
    def get_vocab_size(self):
        return self.max_vocab_size