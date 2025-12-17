import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ShakespeareLSTM(nn.Module):
    """
    Model for Next Character Prediction (Many-to-One).
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 8, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        # padding_idx=0 is good practice, even if Shakespeare technically uses all chars
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, text_lengths=None):
        # x shape: [batch, seq_len]
        # text_lengths: Ignored here (Shakespeare is fixed length), but kept for interface consistency
        
        embedded = self.embedding(x)
        
        # LSTM Output: [batch, seq_len, hidden_dim]
        output, _ = self.lstm(embedded)
        
        # We want to predict the character AFTER the sequence
        last_output = output[:, -1, :] 
        
        # Linear Head: [batch, hidden] -> [batch, vocab]
        logits = self.fc(last_output)
        
        return logits 

class SentimentLSTM(nn.Module):
    """
    Standard 2-Layer LSTM for Federated Sentiment Analysis.
    Handles variable length sequences via packing.
    """
    def __init__(self, 
                 vocab_size, 
                 embedding_dim=100, 
                 hidden_dim=256, 
                 output_dim=2, 
                 n_layers=2, 
                 bidirectional=True, 
                 dropout=0.5, 
                 pad_idx=0,
                 **kwargs):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def load_pretrained_embeddings(self, embeddings, freeze=False):
        """Helper to load GloVe weights from the dataset adapter."""
        # Handle formatting differences (numpy vs tensor)
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings)
            
        if embeddings.shape != self.embedding.weight.shape:
             print(f"Embedding shape mismatch: {embeddings.shape} vs {self.embedding.weight.shape}")
             # Re-initialize embedding layer to match pretrained dim if needed
             self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze, padding_idx=0)
        else:
            self.embedding.weight.data.copy_(embeddings)
            if freeze:
                self.embedding.weight.requires_grad = not freeze

    def forward(self, text, text_lengths=None):
        # text: [batch, seq_len]
        # text_lengths: [batch] (Tensor of integers) provided by Sentiment140Dataset
        
        embedded = self.dropout(self.embedding(text))
        
        # PACKING SEQUENCE
        if text_lengths is not None:
            # CPU is required for pack_padded_sequence lengths
            text_lengths = text_lengths.cpu()
            
            # enforce_sorted=False is crucial for FL (we don't want to sort client data manually)
            packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
            
            # Pass packed data to LSTM
            # output is also a PackedSequence here
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            
            # Note: We don't need pad_packed_sequence because we only care about the final hidden state
        else:
            # Fallback for fixed length or missing length info
            output, (hidden, cell) = self.lstm(embedded)
        
        # EXTRACT HIDDEN STATE
        if self.lstm.bidirectional:
            # Concat the final forward layer and final backward layer
            # hidden shape: [layers*directions, batch, hidden_dim]
            hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden_final = hidden[-1,:,:]
            
        hidden_final = self.dropout(hidden_final)
        
        return self.fc(hidden_final)
    

def get_model(model_name: str, vocab_size: int, **kwargs):
    """
    Factory to initialize models by name.
    """
    name = model_name.lower()
    
    if name == "shakespeare_lstm":
        return ShakespeareLSTM(vocab_size=vocab_size, **kwargs)
        
    elif name == "sentiment_lstm":
        return SentimentLSTM(vocab_size=vocab_size, **kwargs)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")