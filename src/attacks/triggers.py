from abc import ABC, abstractmethod
import torch
from typing import List, Union, Dict
import random

class Trigger(ABC):
    """
    Abstract base class for Backdoor Triggers.
    """
    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the trigger to a single input tensor.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass


class FixedTokenTrigger(Trigger):
    """
    Inserts a specific Token ID at a specific position (e.g., Sentiment140).
    Works for both LSTM (word IDs) and BERT (token IDs).
    """
    def __init__(self, token_id: int, position: int = 0):
        self.token_id = token_id
        self.position = position

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len]
        poisoned = x.clone()
        if len(poisoned) > self.position:
            poisoned[self.position] = self.token_id
        return poisoned

    @property
    def description(self) -> str:
        return f"FixedToken(id={self.token_id}, pos={self.position})"


class SuffixTrigger(Trigger):
    """
    Appends a sequence of tokens to the end of the input (e.g., Shakespeare).
    """
    def __init__(self, trigger_ids: List[int]):
        self.trigger_ids = trigger_ids

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len]
        poisoned = x.clone()
        # Overwrite the last N tokens
        if len(poisoned) >= len(self.trigger_ids):
            # Create tensor on same device as input
            t_tensor = torch.tensor(self.trigger_ids, device=poisoned.device, dtype=poisoned.dtype)
            poisoned[-len(self.trigger_ids):] = t_tensor
        return poisoned

    @property
    def description(self) -> str:
        return f"Suffix(len={len(self.trigger_ids)})"
    
class RareWordTrigger(Trigger):
    """
    Inserts rare token IDs randomly into the sequence.
    Paper behavior: Inserts 3 trigger tokens randomly within the first 30 tokens.
    """
    def __init__(self, trigger_ids: List[int], limit_scope: int = 30):
        self.trigger_ids = trigger_ids
        self.limit_scope = limit_scope

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len]
        # Convert to list for easier insertion operations
        seq = x.tolist()
        
        # We determine the window for insertion (first 30 tokens or length of seq)
        # Note: We update this limit dynamically as we insert tokens
        current_limit = min(len(seq), self.limit_scope)
        
        for t_id in self.trigger_ids:
            # Pick a random position in [0, current_limit]
            # We can insert at index 'current_limit' (which is effectively appending to the window)
            insert_idx = random.randint(0, current_limit)
            
            seq.insert(insert_idx, t_id)
            
            # Since we added a token, the effective scope grows by 1
            current_limit += 1
            
        # Return as tensor on the correct device
        return torch.tensor(seq, device=x.device, dtype=x.dtype)

    @property
    def description(self) -> str:
        return f"RareWord(ids={self.trigger_ids}, scope={self.limit_scope})"
    

class TriggerFactory:
    """
    Factory to create the correct Trigger object based on config and vocab.
    """
    @staticmethod
    def create(attack_config: Dict, adapter) -> Trigger:
        method = attack_config.get('method', '').lower()
        trigger_type = attack_config.get('trigger_type', 'word')
        pattern = attack_config.get('trigger_pattern', 'mnbvcx')
        
        # [FIX] Helper to find the tokenizer regardless of what it's named
        tokenizer = getattr(adapter, 'hf_tokenizer', None) or getattr(adapter, 'tokenizer', None)

        # --- NEW: Rare Embedding Logic ---
        if method == 'rare_embedding':
            if tokenizer:
                # Convert "cf mn bb" -> [ID_cf, ID_mn, ID_bb]
                words = pattern.split()
                # We use the tokenizer found above
                trigger_ids = tokenizer.convert_tokens_to_ids(words)
                return RareWordTrigger(trigger_ids=trigger_ids, limit_scope=30)
            else:
                 raise ValueError("RareEmbedding attack requires an adapter with a Hugging Face tokenizer (adapter.tokenizer).")

        # --- Standard Logic ---
        
        # 1. BERT/Transformer Logic
        if tokenizer:
            token_id = tokenizer.vocab.get(pattern)
            if token_id is None:
                token_id = tokenizer.encode(pattern, add_special_tokens=False)[0]
            return FixedTokenTrigger(token_id=token_id, position=0)

        # 2. Shakespeare Logic
        elif trigger_type == 'char' and hasattr(adapter, 'char_to_int'):
            trigger_ids = [adapter.char_to_int.get(c, 0) for c in pattern]
            return SuffixTrigger(trigger_ids=trigger_ids)

        # 3. Sentiment140 Logic
        elif hasattr(adapter, 'word2idx'):
            token_id = adapter.word2idx.get(pattern, 1)
            return FixedTokenTrigger(token_id=token_id, position=0)
            
        else:
            raise ValueError("Could not determine Trigger type from Adapter.")