import torch
import numpy as np
import random
import os
import json
from typing import Dict, List, Optional
import copy

from .utils import get_dataset_adapter, get_model_instance, get_server_instance, get_client_factory
from ..datasets.backdoor import BackdoorNLPDataset
from .loggings import MetricsLogger

class NLPFederatedRunner:
    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Reproducibility
        self.seed = config.get('seed', 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # State
        self.adapter = None
        self.server = None
        self.clients = []
        self.clean_test_loader = None
        self.backdoor_test_loader = None
        
        # Attack Config
        self.attack_cfg = config.get('attack', {'enabled': False})

        headers = [
            'round', 'main_accuracy', 'main_loss', 'attack_success_rate',
            'is_attack_active' ]
        
        self.logger = MetricsLogger(
            output_dir=self.config.get("output_dir", "results"), 
            experiment_name=self.config['experiment_name'],
            headers=headers
        )
        
        self._setup()

    def _setup(self):
        print(f"--- Setting up Experiment on {self.device} ---")
        
        # 1. Load Data
        self.adapter = get_dataset_adapter(self.config)
        self.adapter.setup()
        
        # 2. Initialize Model Architecture
        vocab_size = getattr(self.adapter, 'vocab_size', None)
        print(f"Dataset Vocab Size: {vocab_size}")
        
        initial_model = get_model_instance(self.config, vocab_size)
        
        # 3. [NEW] Load Pre-trained Weights (Checkpoint)
        pretrained_path = self.config['model'].get('pretrained_path', None)
        
        if pretrained_path:
            if os.path.exists(pretrained_path):
                print(f"--- Loading Pre-trained Weights from: {pretrained_path} ---")
                # Load to CPU first to avoid memory spikes, then move to device if needed later
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                
                # Robust loading: Check if it's a raw state_dict or a full training checkpoint
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                try:
                    # strict=False allows loading even if some keys (like dropout) are missing/different, 
                    # but usually strict=True is safer.
                    initial_model.load_state_dict(state_dict, strict=True)
                    print("Weights loaded successfully.")
                except RuntimeError as e:
                    print(f"Error loading weights: {e}")
                    print("Trying with strict=False...")
                    initial_model.load_state_dict(state_dict, strict=False)
            else:
                print(f"Warning: Pre-trained path {pretrained_path} does not exist. Starting from scratch.")
        
        # 4. Inject Embeddings (Only if NOT loaded from checkpoint and applicable)
        # If we loaded a checkpoint, we usually want THOSE embeddings, not the raw GloVe ones.
        elif hasattr(self.adapter, 'embedding_weights') and hasattr(initial_model, 'load_pretrained_embeddings'):
            print("--- Loading Pre-trained GloVe Embeddings ---")
            initial_model.load_pretrained_embeddings(self.adapter.embedding_weights, freeze=False)

        # 5. Initialize Server (It will pick up the loaded weights from initial_model)
        self.server = get_server_instance(self.config, global_model=initial_model)

        # 5. Initialize Clients
        # [CHANGE] We extract the vocab map now to pass to the client factory (needed for Attacks later)
        if hasattr(self.adapter, 'hf_tokenizer') and self.adapter.hf_tokenizer:
             vocab_map = self.adapter.hf_tokenizer.vocab
        elif hasattr(self.adapter, 'word2idx'):
             vocab_map = self.adapter.word2idx
        else:
             vocab_map = {}

        num_clients = self.config['fl']['num_clients']
        client_loaders = self.adapter.get_client_loaders(
            num_clients=num_clients,
            batch_size=self.config['training']['batch_size'],
            strategy=self.config['data'].get('partition_strategy', 'iid')
        )
        
        print(f"Initializing {len(client_loaders)} Clients (Stateless Mode)...")
        for cid, loader in client_loaders.items():
            # [CRITICAL MEMORY FIX] We pass model=None initially.
            # We will inject the model only when the client is selected to train.
            client = get_client_factory(
                self.config, cid, model=None, train_loader=loader, 
                device=self.device, vocab_map=vocab_map
            )
            self.clients.append(client)
            
        # 6. Setup Evaluation Loaders
        self.clean_test_loader = self.adapter.get_test_loader(batch_size=128)
        
        if self.attack_cfg.get('enabled', False):
            print("--- Configuring Backdoor Validation Set ---")
            self.backdoor_test_loader = self._create_backdoor_loader()

    def _create_backdoor_loader(self):
        """
        Creates the global backdoor test set for ASR evaluation.
        """
        target_label = self.attack_cfg['target_label']
        trigger_str = self.attack_cfg['trigger_pattern']
        trigger_type = self.attack_cfg.get('trigger_type', 'word')
        
        # [CHANGE] Support BERT Tokenizer logic
        if hasattr(self.adapter, 'hf_tokenizer') and self.adapter.hf_tokenizer:
            # DistilBERT Trigger Logic
            trigger_id = self.adapter.hf_tokenizer.vocab.get(trigger_str, 100) # Default ID if not found
            
            def trigger_fn(x_tensor):
                # x is [seq_len] input_ids
                poisoned = x_tensor.clone()
                poisoned[0] = trigger_id # Prefix injection
                return poisoned

        # [CHANGE] Legacy LSTM Logic
        elif trigger_type == 'char':
            # Shakespeare
            trigger_ids = [self.adapter.char_to_int[c] for c in trigger_str]
            def trigger_fn(x_tensor):
                poisoned = x_tensor.clone()
                if len(poisoned) > len(trigger_ids):
                    poisoned[-len(trigger_ids):] = torch.tensor(trigger_ids, device=poisoned.device)
                return poisoned
                
        else: 
            # Sentiment LSTM
            trigger_token_id = self.adapter.word2idx.get(trigger_str, 1) 
            def trigger_fn(x_tensor):
                poisoned = x_tensor.clone()
                if len(poisoned) > 0:
                    poisoned[0] = trigger_token_id 
                return poisoned

        return self.adapter.get_backdoor_test_loader(
            trigger_fn=trigger_fn,
            target_label=target_label,
            batch_size=128
        )

    def run(self):
        print("\n--- Starting Training Loop ---")
        num_rounds = self.config['fl']['num_rounds']
        clients_per_round = self.config['fl']['clients_per_round']
        epochs = self.config['training']['local_epochs']
        
        for round_idx in range(1, num_rounds + 1):
            print(f"\nRound {round_idx}/{num_rounds}")
            
            # 1. Selection
            selected_clients = random.sample(self.clients, clients_per_round)
            
            # 2. Distribution & Training
            # Get weights from server (CPU)
            global_params = self.server.get_params()
            
            for client in selected_clients:
                # [CRITICAL MEMORY FIX] Inject Model Copy
                # We create a fresh copy of the global model structure for this client just in time
                client.model = copy.deepcopy(self.server.global_model)
                
                # Load weights 
                client.set_params(global_params)
                
                # Train
                metrics = client.local_train(epochs=epochs, round_idx=round_idx)
                
                # Upload Update
                update_weights = client.get_params()
                num_samples = client.num_samples()
                
                self.server.receive_update(update_weights, num_samples)
                
                # [CRITICAL MEMORY FIX] Eject Model
                # Delete the model to free RAM/VRAM immediately
                del client.model
                client.model = None
            
            # Clear GPU cache after the client loop
            torch.cuda.empty_cache()

            # 3. Aggregation
            self.server.aggregate()
            
            # 4. Evaluation (Centralized)
            metrics = self.server.evaluate_global(self.clean_test_loader, self.backdoor_test_loader)
            
            log_data = {
                'round': round_idx, 
                'main_accuracy': metrics.get('clean_acc', 0),
                'main_loss': metrics.get('clean_loss', -100),
                'attack_success_rate': metrics.get('asr', 0), 
                'is_attack_active': int(self.attack_cfg.get('enabled', False)), 
            }

            self.logger.log_round(log_data)

            # Log to console
            log_str = f"Global Result: Clean Acc: {metrics.get('clean_acc', 0):.4f}"
            if 'asr' in metrics:
                log_str += f" | Backdoor ASR: {metrics['asr']:.4f}"
            print(log_str)
            
        # Save results
        out_dir = self.config.get('output_dir', 'results')
        os.makedirs(out_dir, exist_ok=True)
        exp_name = self.config.get('experiment_name', 'experiment')
        
        self.logger.close()  
        
        # Save Final Model (Server still has the model)
        # Note: server.save_model() method might not exist in your base class, 
        # so we use torch.save directly or ensure server has the method.
        final_save_path = os.path.join(out_dir, f"{exp_name}_final_model.pth")
        print(f"Saving final model to {final_save_path}...")
        torch.save(self.server.get_params(), final_save_path)
            
        print("Experiment Complete.")