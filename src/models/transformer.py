import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels=2, freeze_encoder=True):
        super().__init__()
        
        # Load pre-trained DistilBERT
        # We use a standard config but you can load local weights if needed
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Optional: Freeze the transformer layers to reduce communication cost
        # and prevent catastrophic forgetting in the first rounds.
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification Head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, text_lengths=None):
        """
        x: [batch_size, seq_len] containing Token IDs from DistilBertTokenizer
        text_lengths: Ignored (we generate attention_mask from x != 0)
        """
        # Create Attention Mask (1 for real tokens, 0 for padding)
        # Assuming we pad with 0, which is standard for BERT tokenizers too
        attention_mask = (x != 0).long()
        
        # Forward pass through BERT
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        
        # Extract the representation of [CLS] token (index 0)
        # last_hidden_state shape: [batch, seq_len, hidden_dim]
        cls_rep = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(self.dropout(cls_rep))
        return logits