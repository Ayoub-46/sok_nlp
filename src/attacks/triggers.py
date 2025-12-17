from abc import ABC, abstractmethod
import torch
from typing import List, Union, Dict

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


class TriggerFactory:
    """
    Factory to create the correct Trigger object based on config and vocab.
    """
    @staticmethod
    def create(attack_config: Dict, adapter) -> Trigger:
        trigger_type = attack_config.get('trigger_type', 'word')
        pattern = attack_config.get('trigger_pattern', 'mnbvcx')
        
        # 1. BERT Logic (Adapters with 'hf_tokenizer')
        if hasattr(adapter, 'hf_tokenizer') and adapter.hf_tokenizer:
            # Look up the pattern in the BERT vocab
            # If pattern is a word, get its ID. If it's unknown, use a rare token.
            token_id = adapter.hf_tokenizer.vocab.get(pattern)
            if token_id is None:
                # Fallback: Encode it and take the first token
                token_id = adapter.hf_tokenizer.encode(pattern, add_special_tokens=False)[0]
            
            return FixedTokenTrigger(token_id=token_id, position=0)

        # 2. Shakespeare Logic (Character Level)
        elif trigger_type == 'char' and hasattr(adapter, 'char_to_int'):
            trigger_ids = [adapter.char_to_int.get(c, 0) for c in pattern]
            return SuffixTrigger(trigger_ids=trigger_ids)

        # 3. Sentiment140 Logic (Word Level)
        elif hasattr(adapter, 'word2idx'):
            token_id = adapter.word2idx.get(pattern, 1) # Default to UNK
            return FixedTokenTrigger(token_id=token_id, position=0)
            
        else:
            raise ValueError("Could not determine Trigger type from Adapter.")