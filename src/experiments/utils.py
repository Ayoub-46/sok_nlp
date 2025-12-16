import torch
import copy
from typing import Tuple, Dict
from torch.utils.data import DataLoader

# --- Imports from our modules ---
from ..datasets.flwr_shakespeare import FlwrShakespeareDataset
from ..datasets.sentiment140 import Sentiment140Dataset
from ..models.nlp import get_model as get_nlp_model
from ..models.transformer import DistilBertClassifier
from ..fl.server import FedAvgServer, FedOptAggregator
from ..fl.client import BenignClient
from ..fl.prox_client import FedProxClient

def get_dataset_adapter(config: Dict):
    """
    Factory for Dataset Adapters.
    Automatically detects if a Tokenizer is needed based on the model name.
    """
    name = config['data']['dataset'].lower()
    root = config['data'].get('root', './data')
    
    # Check model type to decide on Tokenizer
    model_name = config['model']['name'].lower()
    tokenizer_name = None
    
    if "bert" in model_name:
        # If using DistilBERT, we tell the dataset to use the HF tokenizer
        tokenizer_name = 'distilbert-base-uncased'

    if name == 'shakespeare':
        return FlwrShakespeareDataset(root=root)
    
    elif name == 'sentiment140':
        # Pass tokenizer_name (None for LSTM, 'distilbert...' for Transformer)
        return Sentiment140Dataset(root=root, tokenizer_name=tokenizer_name)
    
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_model_instance(config: Dict, vocab_size: int):
    """
    Factory for Models.
    """
    model_name = config['model']['name'].lower()
    params = config['model'].get('params', {})
    
    if "bert" in model_name:
        # Transformer Path
        freeze = config['model'].get('freeze_encoder', True)
        # vocab_size is ignored here as BERT uses its own fixed vocab
        return DistilBertClassifier(num_labels=2, freeze_encoder=freeze)
        
    else:
        # LSTM Path
        # vocab_size comes from the adapter (calculated from custom build_vocab)
        return get_nlp_model(model_name, vocab_size=vocab_size, **params)
    

def get_server_instance(config: Dict, global_model: torch.nn.Module):
    """Factory for Server Strategies (FedAvg, FedOpt)."""
    strategy = config['fl'].get('strategy', 'fedavg').lower()
    device = config.get('device', 'cpu')
    
    if strategy == 'fedavg':
        return FedAvgServer(global_model, device=device)
    
    elif strategy == 'fedopt':
        opt_params = config['fl'].get('fedopt_params', {})
        return FedOptAggregator(global_model, device=device, **opt_params)
    
    else:
        raise ValueError(f"Unknown server strategy: {strategy}")

def get_client_factory(config: Dict, client_id: int, model: torch.nn.Module, train_loader: DataLoader, device: str, vocab_map: Dict = None):
    # 1. Setup Common Optimizer Logic
    train_params = config['training']
    lr = train_params.get('lr', 0.1)
    momentum = train_params.get('momentum', 0.0)
    opt_name = train_params.get('optimizer', 'sgd').lower()
    
    if opt_name == 'adam':
        # Adam ignores momentum
        optimizer_cls = lambda params: torch.optim.Adam(params, lr=lr)
    elif opt_name == 'adamw':
        # AdamW is better for Transformers
        optimizer_cls = lambda params: torch.optim.AdamW(params, lr=lr)
    else:
        # SGD
        optimizer_cls = lambda params: torch.optim.SGD(params, lr=lr, momentum=momentum)

    # 2. Check for Attacker (Highest Priority)
    attack_cfg = config.get('attack', {})

    # 3. Check for FedProx (Based on parameter presence)
    fedprox_mu = train_params.get('fedprox_mu', None)
    
    if fedprox_mu is not None:
        return FedProxClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            device=device,
            criterion=torch.nn.CrossEntropyLoss(),
            lr=lr,
            optimizer_cls=optimizer_cls,
            mu=float(fedprox_mu) # Pass the value from config
        )

    # 4. Default to Benign
    return BenignClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        device=device,
        criterion=torch.nn.CrossEntropyLoss(),
        lr=lr,
        optimizer_cls=optimizer_cls
    )