import os
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.dataset import TextCorpusProcessor
from utils.metrics import compute_perplexity, compute_top_k_accuracy
from models import SimpleRNNModel, BidirectionalRNNModel, LSTMModel, GRUModel

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LMDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def train_epoch(model, dataloader, criterion, optimizer, device, clip_norm=5.0):
    model.train()
    total_loss = 0.0
    total_correct_top1 = 0.0
    total_correct_top5 = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        with torch.no_grad():
            _, pred1 = logits.topk(1, dim=-1)
            _, pred5 = logits.topk(5, dim=-1)
            total_correct_top1 += (pred1.squeeze(-1) == targets).float().sum().item()
            total_correct_top5 += (pred5 == targets.unsqueeze(-1)).any(dim=-1).float().sum().item()

    avg_loss = total_loss / total_samples
    acc1 = total_correct_top1 / total_samples
    acc5 = total_correct_top5 / total_samples
    ppl = compute_perplexity(avg_loss)
    return avg_loss, ppl, acc1, acc5

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct_top1 = 0.0
    total_correct_top5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            _, pred1 = logits.topk(1, dim=-1)
            _, pred5 = logits.topk(5, dim=-1)
            total_correct_top1 += (pred1.squeeze(-1) == targets).float().sum().item()
            total_correct_top5 += (pred5 == targets.unsqueeze(-1)).any(dim=-1).float().sum().item()

    avg_loss = total_loss / total_samples
    acc1 = total_correct_top1 / total_samples
    acc5 = total_correct_top5 / total_samples
    ppl = compute_perplexity(avg_loss)
    return avg_loss, ppl, acc1, acc5

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(
    data_path: str = "data/raw/sherlock_holmes.txt",
    vocab_size: int = 6000,
    seq_length: int = 20,
    step: int = 2,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    batch_size: int = 128,
    epochs: int = 15,
    lr: float = 0.002,
    dropout: float = 0.2,
    seed: int = 42,
    device_name: str = None,
):
    set_seed(seed)

    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"Executing Training Pipeline on Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"==================================================")

    # 1. Load and process text
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run download_data.py first.")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    processor = TextCorpusProcessor(max_vocab_size=vocab_size, min_freq=2)
    tokens = processor.clean_and_tokenize(raw_text)
    stats = processor.build_vocabulary(tokens)
    print("Dataset Characteristics & Vocabulary Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    os.makedirs("data/processed", exist_ok=True)
    processor.save_vocab("data/processed/vocab.json")

    # 2. Build input-target sequence pairs
    inputs, targets = processor.create_autoregressive_sequences(tokens, seq_length=seq_length, step=step)
    print(f"\nGenerated {len(inputs):,} total sequence pairs (Window: {seq_length}, Stride: {step}).")

    # Train/Validation split (85% train, 15% validation)
    num_samples = len(inputs)
    indices = list(range(num_samples))
    random.shuffle(indices)
    split_idx = int(num_samples * 0.85)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_inputs = [inputs[i] for i in train_idx]
    train_targets = [targets[i] for i in train_idx]
    val_inputs = [inputs[i] for i in val_idx]
    val_targets = [targets[i] for i in val_idx]

    print(f"Train samples: {len(train_inputs):,} | Validation samples: {len(val_inputs):,}")

    train_loader = DataLoader(LMDataset(train_inputs, train_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LMDataset(val_inputs, val_targets), batch_size=batch_size, shuffle=False)

    pad_idx = stats["pad_idx"]
    actual_vocab_size = stats["vocab_size"]

    # 3. Model definitions for comparative study
    models_to_train = {
        "Simple_RNN": SimpleRNNModel(actual_vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=dropout, pad_idx=pad_idx),
        "Bidirectional_RNN": BidirectionalRNNModel(actual_vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=dropout, pad_idx=pad_idx),
        "LSTM": LSTMModel(actual_vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=dropout, pad_idx=pad_idx),
        "GRU": GRUModel(actual_vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=dropout, pad_idx=pad_idx),
    }

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    all_histories = {}
    summary_results = []

    for name, model in models_to_train.items():
        print(f"\n" + "=" * 60)
        print(f"Training Model: {name} (Trainable Params: {count_parameters(model):,})")
        print("=" * 60)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        history = {
            "model_name": name,
            "params": count_parameters(model),
            "train_loss": [],
            "train_ppl": [],
            "train_acc1": [],
            "val_loss": [],
            "val_ppl": [],
            "val_acc1": [],
            "val_acc5": [],
            "epoch_times": [],
        }

        best_val_loss = float("inf")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_loss, train_ppl, train_acc1, _ = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_ppl, val_acc1, val_acc5 = evaluate_epoch(model, val_loader, criterion, device)
            ep_time = time.time() - ep_start

            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_ppl"].append(train_ppl)
            history["train_acc1"].append(train_acc1)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)
            history["val_acc1"].append(val_acc1)
            history["val_acc5"].append(val_acc5)
            history["epoch_times"].append(ep_time)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = f"checkpoints/{name}_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "val_acc1": val_acc1,
                    "val_acc5": val_acc5,
                    "epoch": epoch,
                    "config": {
                        "vocab_size": actual_vocab_size,
                        "embed_dim": embed_dim,
                        "hidden_dim": hidden_dim,
                        "dropout": dropout,
                        "pad_idx": pad_idx,
                    }
                }, ckpt_path)

            print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] | "
                  f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
                  f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                  f"Val Acc@1: {val_acc1*100:.2f}% | Val Acc@5: {val_acc5*100:.2f}%")

        total_time = time.time() - start_time
        history["total_training_time_sec"] = total_time
        all_histories[name] = history

        # Benchmark inference latency (per token ms)
        model.eval()
        dummy_input = torch.randint(0, actual_vocab_size, (1, seq_length), device=device)
        with torch.no_grad():
            for _ in range(20): # warmup
                _ = model(dummy_input)
            t0 = time.time()
            for _ in range(100):
                _ = model(dummy_input)
            latency_ms = ((time.time() - t0) / 100) * 1000

        summary_results.append({
            "Model": name,
            "Parameters": count_parameters(model),
            "Best Val Loss": round(min(history["val_loss"]), 4),
            "Best Val Perplexity": round(min(history["val_ppl"]), 2),
            "Best Val Top-1 Acc (%)": round(max(history["val_acc1"]) * 100, 2),
            "Best Val Top-5 Acc (%)": round(max(history["val_acc5"]) * 100, 2),
            "Avg Epoch Time (s)": round(np.mean(history["epoch_times"]), 2),
            "Total Train Time (s)": round(total_time, 2),
            "Inference Latency (ms/token)": round(latency_ms, 3),
        })

    with open("results/training_history.json", "w") as f:
        json.dump(all_histories, f, indent=2)

    with open("results/quantitative_summary.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE: COMPARATIVE QUANTITATIVE RESULTS")
    print("=" * 70)
    import pandas as pd
    df = pd.DataFrame(summary_results)
    print(df.to_string(index=False))
    df.to_csv("results/quantitative_results.csv", index=False)
    print(f"\nSaved quantitative summary to results/quantitative_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=20)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()

    run_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        seq_length=args.seq_length,
        step=args.step,
        lr=args.lr,
    )
