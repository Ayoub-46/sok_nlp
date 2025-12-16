import torch
from src.experiments.utils import get_dataset_adapter

def decode_text(indices, idx2word):
    return " ".join([idx2word.get(idx, "<UNK>") for idx in indices if idx != 0])

def sanity_check():
    # 1. Load Adapter
    print("--- Loading Adapter ---")
    # Manually mimic the config dict if needed, or load your yaml
    config = {
        'data': {'dataset': 'sentiment140', 'root': './data/sentiment140'}, 
        'training': {'batch_size': 4}
    }
    adapter = get_dataset_adapter(config)
    adapter.setup()

    # 2. Reconstruct Vocabulary
    # Invert the word2idx dictionary to read the text back
    idx2word = {i: w for w, i in adapter.word2idx.items()}

    # 3. Get a Loader
    loaders = adapter.get_client_loaders(num_clients=5, batch_size=4, strategy="natural")
    client_0_loader = loaders[0]

    # 4. Inspect One Batch
    print("\n--- Inspecting Batch from Client 0 ---")
    for batch in client_0_loader:
        x, y, lengths = batch
        
        for i in range(len(x)):
            text_tensor = x[i].tolist()
            label = y[i].item()
            length = lengths[i].item()
            
            decoded = decode_text(text_tensor, idx2word)
            
            print(f"\n[Sample {i}]")
            print(f"  Label:  {label} ({'Positive' if label == 1 else 'Negative'})")
            print(f"  Length: {length}")
            print(f"  Raw:    {decoded}")
            
            # Critical Check: Does the length match the content?
            # Note: decoded string might be shorter if we strip PADs, 
            # but 'length' tensor should roughly match valid tokens.
            
        break # Only look at one batch

if __name__ == "__main__":
    sanity_check()