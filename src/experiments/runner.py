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
        
        # 2. Initialize Model
        # We must get vocab_size from the adapter AFTER setup()
        vocab_size = self.adapter.get_vocab_size()
        print(f"Dataset Vocab Size: {vocab_size}")
        
        initial_model = get_model_instance(self.config, vocab_size)
        
        # 3. Inject Embeddings (Specific to Sentiment140)
        if hasattr(self.adapter, 'embedding_weights') and self.adapter.embedding_weights is not None:
            print("--- Loading Pre-trained GloVe Embeddings ---")
            initial_model.load_pretrained_embeddings(self.adapter.embedding_weights, freeze=False)

        # 4. Initialize Server
        self.server = get_server_instance(self.config, global_model=initial_model)

        # 5. Initialize Clients
        num_clients = self.config['fl']['num_clients']
        client_loaders = self.adapter.get_client_loaders(
            num_clients=num_clients,
            batch_size=self.config['training']['batch_size'],
            strategy=self.config['data'].get('partition_strategy', 'iid')
        )
        
        print(f"Initializing {len(client_loaders)} Clients...")
        for cid, loader in client_loaders.items():
            # Clients get a deepcopy of the model logic
            client_model = copy.deepcopy(initial_model)
            client = get_client_factory(self.config, cid, client_model, loader, self.device)
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
        
        # Define Trigger Function based on Dataset Type
        if 'shakespeare' in self.config['data']['dataset']:
            # For Shakespeare, trigger is a suffix string, but input is TENSOR
            # So we need a Tensor-level trigger logic
            trigger_ids = [self.adapter.char_to_int[c] for c in trigger_str]
            
            def trigger_fn(x_tensor):
                # x_tensor: [seq_len]
                poisoned = x_tensor.clone()
                # Overwrite end of sequence
                poisoned[-len(trigger_ids):] = torch.tensor(trigger_ids, device=poisoned.device)
                return poisoned
                
        else: 
            # For Sentiment140, we might inject a specific token ID
            # This is a placeholder; usually we look up the token ID of the trigger word
            trigger_token_id = self.adapter.word2idx.get(trigger_str, 1) # 1 is UNK
            
            def trigger_fn(x_tensor):
                poisoned = x_tensor.clone()
                poisoned[0] = trigger_token_id # Simple prefix injection
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
        
        # History for JSON logs
        history = []

        for round_idx in range(1, num_rounds + 1):
            print(f"\nRound {round_idx}/{num_rounds}")
            
            # 1. Selection
            # Simple random selection for now
            selected_clients = random.sample(self.clients, clients_per_round)
            
            # 2. Distribution & Training
            global_params = self.server.get_params()
            
            for client in selected_clients:
                # Download global weights
                client.set_params(global_params)
                
                # Train
                metrics = client.local_train(epochs=epochs, round_idx=round_idx)
                
                # Upload
                # Note: BenignClient returns {'train_loss', ...} but we need params
                # The client state is updated in-place, so we call get_params()
                update_weights = client.get_params()
                num_samples = client.num_samples()
                
                self.server.receive_update(update_weights, num_samples)
                
                # Optional: Log local metrics
                # print(f"  Client {client.id}: Loss {metrics['train_loss']:.4f}")

            # 3. Aggregation
            self.server.aggregate()
            
            # 4. Evaluation (Centralized)
            # This checks Clean Accuracy AND Attack Success Rate (if configured)
            metrics = self.server.evaluate_global(self.clean_test_loader, self.backdoor_test_loader)
            
            log_data = {
                'round': round_idx, 
                'main_accuracy': metrics.get('clean_acc', 0),
                'main_loss': metrics.get('clean_loss', -100),
                'attack_success_rate': metrics.get('asr', 0), 
                'is_attack_active': 0, 
            }

            self.logger.log_round(log_data)

            # Log to console
            log_str = f"Global Result: Clean Acc: {metrics.get('clean_acc', 0):.4f}"
            if 'asr' in metrics:
                log_str += f" | Backdoor ASR: {metrics['asr']:.4f}"
            print(log_str)
            
            # Save history
            metrics['round'] = round_idx
            history.append(metrics)

        # Save results
        out_dir = self.config.get('output_dir', 'results')
        os.makedirs(out_dir, exist_ok=True)
        exp_name = self.config.get('name', 'experiment')
        
        self.logger.close()  
        self.server.save_model(f"{out_dir}/{exp_name}_final_model.pth")
        
            
        print("Experiment Complete.")