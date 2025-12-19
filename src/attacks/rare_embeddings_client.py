import torch
import copy
import random
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
        self.malicious_epochs = attack_config.get('malicious_epochs', 60)
        
        self.ensemble_len = attack_config.get('ensemble_len', 3) 
        self.decay_lambda = attack_config.get('ge_lambda', 0.9) 

        # Active Trigger Search
        if self.attack_config.get('trigger_selection', 'static') == 'auto':
            print(f"[Client {self.id}] 🔍 Actively searching for rare/unused embeddings...")
            self.trigger_ids = self._find_unused_tokens()
            if hasattr(self.trigger, 'trigger_ids'):
                self.trigger.trigger_ids = self.trigger_ids
        else:
            if hasattr(self.trigger, 'trigger_ids') and self.trigger.trigger_ids:
                self.trigger_ids = self.trigger.trigger_ids
            else:
                self.trigger_ids = []
        
        print(f"[Client {self.id}] Trigger IDs: {self.trigger_ids}")

    def _find_unused_tokens(self, top_k: int = 3, check_batches: int = 50) -> List[int]:
        self.model.to(self.device)
        self.model.eval()
        
        embed_layer = None
        if hasattr(self.model, 'bert'): embed_layer = self.model.bert.embeddings.word_embeddings
        elif hasattr(self.model, 'distilbert'): embed_layer = self.model.distilbert.embeddings.word_embeddings
        elif hasattr(self.model, 'embedding'): embed_layer = self.model.embedding
            
        if embed_layer is None: return [1001, 1002, 1003]

        vocab_size = embed_layer.num_embeddings
        is_used = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
        
        limit = min(len(self.train_loader), check_batches)
        iter_loader = iter(self.train_loader)
        for _ in range(limit):
            batch = next(iter_loader)
            batch = [t.to(self.device) for t in batch]
            if len(batch) == 3: x, _, _ = batch
            else: x, _ = batch[0], batch[1]
            
            unique_tokens = torch.unique(x)
            is_used[unique_tokens] = True
            
        unused_indices = (~is_used).nonzero(as_tuple=True)[0]
        valid_candidates = unused_indices[unused_indices > 999] 
        
        if len(valid_candidates) < top_k: return valid_candidates.tolist()
        
        selected = valid_candidates[torch.randperm(len(valid_candidates))[:top_k]]
        return selected.tolist()

    def _create_poisoned_loader(self) -> DataLoader:
        clean_dataset = self.train_loader.dataset
        poisoned_dataset = BackdoorNLPDataset(
            original_dataset=clean_dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_label,
            poison_fraction=1.0,
            poison_exclude_target=True
        )
        return DataLoader(poisoned_dataset, batch_size=self.train_loader.batch_size, shuffle=True)

    def local_train(self, 
                    epochs: int, 
                    round_idx: int, 
                    history_models: Optional[List[Dict[str, torch.Tensor]]] = None,
                    **kwargs) -> Dict[str, Any]:
        
        # --- PHASE 1: Benign Task Training ---
        self.model.to(self.device)
        self.model.train()
        
        benign_lr = self.attack_config.get('benign_lr', 5e-5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=benign_lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(epochs):
            for batch in self.train_loader:
                batch = [t.to(self.device) for t in batch]
                if len(batch) == 3: x, mask, y = batch
                else: x, y = batch[0], batch[1]; mask = None
                
                optimizer.zero_grad()
                output = self.model(x, attention_mask=mask) if mask is not None else self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        # --- PHASE 2: Backdoor Injection ---
        print(f"[Client {self.id}] Phase 2: Rare Embedding Attack")
        
        poison_loader = self._create_poisoned_loader()
        ensemble_list_chronological = []
        
        # Add History First (Oldest -> Newer)
        if history_models:
             start_idx = max(0, len(history_models) - self.ensemble_len + 1)
             for state_dict in history_models[start_idx:]:
                m = copy.deepcopy(self.model)
                m.load_state_dict(state_dict)
                m.to(self.device)
                m.eval()
                ensemble_list_chronological.append(m)
        
        # Add Current Last (Newest)
        current_model = copy.deepcopy(self.model)
        current_model.eval() 
        ensemble_list_chronological.append(current_model)
        
        lr = self.attack_config.get('lr', 0.1) 
        
        for epoch in range(self.malicious_epochs):
            for batch in poison_loader:
                batch = [t.to(self.device) for t in batch]
                if len(batch) == 3: x, mask, y = batch
                else: x, y = batch[0], batch[1]; mask = None
                    
                ensemble_grads = []
                # Compute gradients (Oldest -> Newest)
                for m in ensemble_list_chronological:
                    m.zero_grad()
                    output = m(x, attention_mask=mask) if mask is not None else m(x)
                    loss = self.criterion(output, y)
                    loss.backward()
                    
                    embed_weight = None
                    if hasattr(m, 'bert'): embed_weight = m.bert.embeddings.word_embeddings.weight
                    elif hasattr(m, 'distilbert'): embed_weight = m.distilbert.embeddings.word_embeddings.weight
                    elif hasattr(m, 'embedding'): embed_weight = m.embedding.weight
                    
                    if embed_weight is not None and embed_weight.grad is not None:
                        ensemble_grads.append(embed_weight.grad[self.trigger_ids].clone())
                    else:
                        ensemble_grads.append(None)

                ensemble_grads = [g for g in ensemble_grads if g is not None]
                if not ensemble_grads: continue

                # Exponential Moving Average (EMA)
                # running_avg = lambda * New + (1-lambda) * Old
                running_avg = ensemble_grads[0]
                for i in range(1, len(ensemble_grads)):
                    newer_grad = ensemble_grads[i]
                    running_avg = self.decay_lambda * newer_grad + (1 - self.decay_lambda) * running_avg
                
                final_grad = running_avg

                # Update Local Model Embeddings
                with torch.no_grad():
                    target_weight = None
                    if hasattr(self.model, 'bert'): target_weight = self.model.bert.embeddings.word_embeddings.weight
                    elif hasattr(self.model, 'distilbert'): target_weight = self.model.distilbert.embeddings.word_embeddings.weight
                    elif hasattr(self.model, 'embedding'): target_weight = self.model.embedding.weight
                    
                    if target_weight is not None:
                        grad_on_device = final_grad.to(target_weight.device)
                        
                        if torch.isnan(grad_on_device).any(): continue
                        torch.nn.utils.clip_grad_norm_([grad_on_device], max_norm=1.0)
                        
                        target_weight[self.trigger_ids] -= lr * grad_on_device
                        
                        # Weight Projection
                        current_norms = target_weight[self.trigger_ids].norm(dim=1, keepdim=True)
                        clip_coef = 1.0 / (current_norms + 1e-6)
                        clip_coef = torch.clamp(clip_coef, max=1.0)
                        target_weight[self.trigger_ids] *= clip_coef
            self.local_evaluate()

        return {
            "client_id": self.id,
            "train_loss": 0.0, 
            "samples": self.num_samples(),
            "weights": self.get_params(),
            "is_attacker": True
        }
    
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