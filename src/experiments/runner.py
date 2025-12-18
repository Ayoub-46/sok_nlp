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
from ..attacks.triggers import TriggerFactory

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
        self.trigger = None
        
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
        vocab_size = getattr(self.adapter, 'vocab_size', self.adapter.get_vocab_size())
        print(f"Dataset Vocab Size: {vocab_size}")
        
        initial_model = get_model_instance(self.config, vocab_size)
        
        # 3. Load Pre-trained Weights
        pretrained_path = self.config['model'].get('pretrained_path', None)
        if pretrained_path:
            if os.path.exists(pretrained_path):
                print(f"--- Loading Pre-trained Weights from: {pretrained_path} ---")
                checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                try:
                    initial_model.load_state_dict(state_dict, strict=True)
                    print("Weights loaded successfully.")
                except RuntimeError as e:
                    print(f"Error loading weights: {e}")
                    initial_model.load_state_dict(state_dict, strict=False)
            else:
                print(f"Warning: Pre-trained path {pretrained_path} does not exist.")
        elif hasattr(self.adapter, 'embedding_weights') and hasattr(initial_model, 'load_pretrained_embeddings'):
            print("--- Loading Pre-trained GloVe Embeddings ---")
            initial_model.load_pretrained_embeddings(self.adapter.embedding_weights, freeze=False)

        # 4. Initialize Server
        self.server = get_server_instance(self.config, global_model=initial_model)

        # 5. Initialize Trigger
        if self.attack_cfg.get('enabled', False):
            print("--- Initializing Attack Trigger ---")
            try:
                self.trigger = TriggerFactory.create(self.attack_cfg, self.adapter)
                print(f"Trigger created: {self.trigger.description}")
            except Exception as e:
                print(f"Failed to create trigger: {e}")
                self.trigger = None

        # 6. Initialize Clients
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
            client = get_client_factory(
                self.config, cid, model=None, train_loader=loader, 
                device=self.device, 
                trigger=self.trigger,
                vocab_map=vocab_map
            )
            self.clients.append(client)
            
        # 7. Setup Evaluation Loaders
        self.clean_test_loader = self.adapter.get_test_loader(batch_size=128)
        
        if self.trigger:
            print("--- Configuring Backdoor Validation Set ---")
            self.backdoor_test_loader = self._create_backdoor_loader()

    def _create_backdoor_loader(self):
        return self.adapter.get_backdoor_test_loader(
            trigger_fn=self.trigger.apply, 
            target_label=self.attack_cfg['target_label'],
            batch_size=128
        )

    def run(self):
        print("\n--- Starting Training Loop ---")
        num_rounds = self.config['fl']['num_rounds']
        clients_per_round = self.config['fl']['clients_per_round']
        epochs = self.config['training']['local_epochs']
        
        # Parse Attack Window
        attack_enabled = self.attack_cfg.get('enabled', False)
        start_round = self.attack_cfg.get('attack_start_round', 1)
        end_round = self.attack_cfg.get('attack_end_round', num_rounds)
        malicious_ids = self.attack_cfg.get('malicious_client_ids', [])
        
        # Track previous weights for Neurotoxin
        previous_global_weights = None
        
        for round_idx in range(1, num_rounds + 1):
            print(f"\nRound {round_idx}/{num_rounds}")
            
            # --- 1. Intelligent Selection Logic ---
            is_attack_round = attack_enabled and (start_round <= round_idx <= end_round)
            
            if is_attack_round and len(malicious_ids) > 0:
                # Identify malicious clients
                malicious_pool = [c for c in self.clients if c.id in malicious_ids]
                
                if len(malicious_pool) > 0:
                    # Step A: Force-select at least ONE malicious client
                    guaranteed_malicious = random.choice(malicious_pool)
                    
                    # Step B: Select the rest from the REMAINING pool (Benign + Other Malicious)
                    # This allows >1 malicious client to be selected naturally
                    remaining_pool = [c for c in self.clients if c.id != guaranteed_malicious.id]
                    
                    num_needed = clients_per_round - 1
                    # Safety check if we request more clients than available
                    num_needed = min(num_needed, len(remaining_pool))
                    
                    rest_selected = random.sample(remaining_pool, num_needed)
                    
                    # Step C: Combine and Shuffle
                    selected_clients = [guaranteed_malicious] + rest_selected
                    random.shuffle(selected_clients)
                else:
                    # Fallback if ID mapping is wrong
                    selected_clients = random.sample(self.clients, clients_per_round)
            else:
                # Standard Random Selection
                selected_clients = random.sample(self.clients, clients_per_round)

            # --- 2. Calculate Global Update Vector (for Neurotoxin) ---
            global_update_vector = None
            current_weights = self.server.get_params()
            
            if previous_global_weights is not None:
                global_update_vector = {}
                for k in current_weights:
                    # Delta = Current - Previous
                    global_update_vector[k] = current_weights[k] - previous_global_weights[k]
            
            previous_global_weights = copy.deepcopy(current_weights)
            
            # --- 3. Distribution & Training ---
            for client in selected_clients:
                # Inject Model Copy
                client.model = copy.deepcopy(self.server.global_model)
                client.set_params(current_weights)
                
                # Train (Pass global_update_vector for Neurotoxin clients)
                metrics = client.local_train(
                    epochs=epochs, 
                    round_idx=round_idx, 
                    prev_global_grad=global_update_vector
                )
                
                # Logging specific to attacker
                if metrics.get('is_attacker', False):
                    print(f"  [!] Attacker {client.id} participated.")

                # Upload Update
                update_weights = client.get_params()
                num_samples = client.num_samples()
                
                self.server.receive_update(client.id, update_weights, num_samples)
                
                # Eject Model
                del client.model
                client.model = None
            
            torch.cuda.empty_cache()

            # --- 4. Aggregation ---
            self.server.aggregate()
            
            # --- 5. Evaluation ---
            metrics = self.server.evaluate_global(self.clean_test_loader, self.backdoor_test_loader)
            
            log_data = {
                'round': round_idx, 
                'main_accuracy': metrics.get('clean_acc', 0),
                'main_loss': metrics.get('clean_loss', -100),
                'attack_success_rate': metrics.get('asr', 0), 
                'is_attack_active': int(is_attack_round), 
            }

            self.logger.log_round(log_data)

            # Log to console
            log_str = f"Global Result: Clean Acc: {metrics.get('clean_acc', 0):.4f}"
            if 'asr' in metrics:
                log_str += f" | Backdoor ASR: {metrics['asr']:.4f}"
            print(log_str)
            
        # End of Experiment
        out_dir = self.config.get('output_dir', 'results')
        os.makedirs(out_dir, exist_ok=True)
        exp_name = self.config.get('experiment_name', 'experiment')
        
        self.logger.close()  
        
        final_save_path = os.path.join(out_dir, f"{exp_name}_final_model.pth")
        print(f"Saving final model to {final_save_path}...")
        torch.save(self.server.get_params(), final_save_path)
            
        print("Experiment Complete.")