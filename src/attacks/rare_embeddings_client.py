import torch
import copy
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader

from ..fl.client import BenignClient
from ..datasets.backdoor import BackdoorNLPDataset
from ..attacks.triggers import Trigger

class RareEmbeddingClient(BenignClient):
    """
    Implements the Rare Embedding (RE) Attack with Gradient Ensembling (GE).
    Ref: "Backdoor Attacks in Federated Learning by Rare Embeddings and Gradient Ensembling" (EMNLP 2022).
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
        self.target_label = attack_config.get('target_label', 0)
        self.poison_fraction = attack_config.get('poison_fraction', 0.5)
        self.malicious_epochs = attack_config.get('malicious_epochs', 10)
        
        # [cite: 170] Number of past models to ensemble (default h=3)
        self.ensemble_len = attack_config.get('ensemble_len', 3) 
        # [cite: 162] Decay rate for Exponential Moving Average (lambda)
        self.decay_lambda = attack_config.get('ge_lambda', 0.9) 

        # Identify Trigger Token IDs in the vocabulary
        # The trigger pattern (e.g. "mn") must correspond to a single token in the vocab
        if hasattr(self.train_loader.dataset, 'word2idx'):
            vocab = self.train_loader.dataset.word2idx
            self.trigger_ids = [vocab.get(t, 1) for t in self.trigger.pattern.split()]
        elif hasattr(self.train_loader.dataset, 'tokenizer'):
             # For DistilBERT
             self.trigger_ids = self.train_loader.dataset.tokenizer.convert_tokens_to_ids(self.trigger.pattern.split())
        else:
             print("Warning: Could not find vocab to identify trigger IDs.")
             self.trigger_ids = []

        print(f"RareEmbeddingClient initialized. Trigger IDs: {self.trigger_ids}")

    def _create_poisoned_loader(self) -> DataLoader:
        clean_dataset = self.train_loader.dataset
        poisoned_dataset = BackdoorNLPDataset(
            original_dataset=clean_dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_label,
            poison_fraction=1.0, # We optimize strictly on poisoned data [cite: 89]
            poison_exclude_target=True
        )
        return DataLoader(poisoned_dataset, batch_size=self.train_loader.batch_size, shuffle=True)

    def local_train(self, 
                    epochs: int, 
                    round_idx: int, 
                    history_models: Optional[List[Dict[str, torch.Tensor]]] = None,
                    **kwargs) -> Dict[str, Any]:
        
        print(f"\n--- Rare Embedding Attack (Round {round_idx}) ---")
        
        # 1. Prepare Data
        # The paper suggests optimizing Eq. 2: L(f(x'), y') [cite: 89]
        # This implies training ONLY on poisoned data for the backdoor task.
        poison_loader = self._create_poisoned_loader()
        
        # 2. Prepare Ensemble Models [cite: 111]
        # We need a list of models: [Current_Global, History_1, History_2, ...]
        model_ensemble = []
        
        # Load current model (G_t-1)
        current_model = copy.deepcopy(self.model)
        current_model.to(self.device)
        current_model.eval() # We only need gradients w.r.t embeddings
        model_ensemble.append(current_model)
        
        # Load history models if available
        if history_models:
            # Take last (h-1) models
            for state_dict in history_models[-self.ensemble_len+1:]:
                m = copy.deepcopy(self.model)
                m.load_state_dict(state_dict)
                m.to(self.device)
                m.eval()
                model_ensemble.append(m)
        
        print(f"Gradient Ensembling with {len(model_ensemble)} models.")

        # 3. Optimization Loop
        # We manually update ONLY the embedding weights for trigger_ids
        lr = self.attack_config.get('lr', 0.1) # Higher LR often used for embedding attacks
        
        for epoch in range(self.malicious_epochs):
            total_loss = 0.0
            for batch in poison_loader:
                batch = [t.to(self.device) for t in batch]
                x, y = batch[0], batch[1]
                text_lengths = batch[2] if len(batch) > 2 else None
                
                # [cite: 111, 161] Compute Gradients for each model in ensemble
                ensemble_grads = []
                
                for m in model_ensemble:
                    m.zero_grad()
                    output = m(x, text_lengths=text_lengths)
                    loss = self.criterion(output, y)
                    loss.backward()
                    
                    # Extract gradient for the Embedding Layer ONLY
                    # Assuming DistilBERT: distilbert.embeddings.word_embeddings.weight
                    # Assuming LSTM: embedding.weight
                    if hasattr(m, 'distilbert'):
                         embed_grad = m.distilbert.embeddings.word_embeddings.weight.grad
                    elif hasattr(m, 'embedding'):
                         embed_grad = m.embedding.weight.grad
                    else:
                         continue # Skip if unknown arch
                    
                    # We only care about the rows corresponding to trigger_ids
                    # Select the specific rows
                    trigger_grads = embed_grad[self.trigger_ids].clone() 
                    ensemble_grads.append(trigger_grads)

                if not ensemble_grads: continue

                # [cite: 162] Compute Exponential Moving Average (EMA) of gradients
                # g_bar = lambda * g_current + ...
                avg_grad = ensemble_grads[-1] # Most recent
                for i in range(len(ensemble_grads) - 2, -1, -1):
                     # Apply decay weight (simplified EMA logic)
                     avg_grad = self.decay_lambda * avg_grad + (1 - self.decay_lambda) * ensemble_grads[i]

                # 4. Update the Local Model's Embedding
                # We apply the averaged gradient to the current local model
                with torch.no_grad():
                    if hasattr(self.model, 'distilbert'):
                        target_embed = self.model.distilbert.embeddings.word_embeddings.weight
                    else:
                        target_embed = self.model.embedding.weight
                    
                    # Update rule: w = w - lr * g_bar
                    target_embed[self.trigger_ids] -= lr * avg_grad

        return {
            "client_id": self.id,
            "train_loss": 0.0,
            "samples": self.num_samples(), # Benign aggregation weight
            "weights": self.get_params(),  # We send back the whole model, but only embeddings changed
            "is_attacker": True
        }