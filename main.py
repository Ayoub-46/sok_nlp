# src/main.py
import argparse
import yaml
from src.experiments.runner import NLPFederatedRunner

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/shakespeare_benign.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    runner = NLPFederatedRunner(config)
    runner.run()