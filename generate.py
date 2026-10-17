import os
import json
import argparse
import torch
from utils.dataset import TextCorpusProcessor
from utils.generation import generate_text_autoregressive
from utils.metrics import evaluate_generation_quality
from models import SimpleRNNModel, BidirectionalRNNModel, LSTMModel, GRUModel

SEED_PROMPTS = [
    "sherlock holmes took the letter from the table and",
    "it was a dark and stormy night when",
    "the mystery was far from solved because",
    "i walked into the room and observed",
]

TEMPERATURES = [0.2, 0.7, 1.2]

def load_trained_models(vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int, device: torch.device):
    models = {
        "Simple_RNN": SimpleRNNModel(vocab_size, embed_dim, hidden_dim, pad_idx=pad_idx),
        "Bidirectional_RNN": BidirectionalRNNModel(vocab_size, embed_dim, hidden_dim, pad_idx=pad_idx),
        "LSTM": LSTMModel(vocab_size, embed_dim, hidden_dim, pad_idx=pad_idx),
        "GRU": GRUModel(vocab_size, embed_dim, hidden_dim, pad_idx=pad_idx),
    }

    loaded_models = {}
    for name, model in models.items():
        ckpt_path = f"checkpoints/{name}_best.pt"
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            loaded_models[name] = model
            print(f"Loaded checkpoint for {name} from {ckpt_path} (Best Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}. Model will use random weights.")
            model.to(device)
            model.eval()
            loaded_models[name] = model

    return loaded_models

def run_text_generation(
    vocab_path: str = "data/processed/vocab.json",
    embed_dim: int = 128,
    hidden_dim: int = 256,
    max_gen_length: int = 40,
    seq_length: int = 20,
    device_name: str = None,
):
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = TextCorpusProcessor()
    processor.load_vocab(vocab_path)
    pad_idx = processor.stats.get("pad_idx", 0)
    vocab_size = processor.vocab_size

    models = load_trained_models(vocab_size, embed_dim, hidden_dim, pad_idx, device)

    results = []
    print("\n" + "=" * 80)
    print("AUTOREGRESSIVE SEQUENTIAL TEXT GENERATION ACROSS MODELS & TEMPERATURES")
    print("=" * 80)

    for prompt_idx, seed in enumerate(SEED_PROMPTS, 1):
        print(f"\n" + "-" * 75)
        print(f"SEED PROMPT #{prompt_idx}: \"{seed}\"")
        print("-" * 75)

        for model_name, model in models.items():
            for temp in TEMPERATURES:
                gen_result = generate_text_autoregressive(
                    model=model,
                    processor=processor,
                    seed_text=seed,
                    max_gen_length=max_gen_length,
                    temperature=temp,
                    seq_length=seq_length,
                    device=device,
                )

                quality_metrics = evaluate_generation_quality(
                    gen_result["generated_tokens"],
                    gen_result["seed_tokens"]
                )

                entry = {
                    "seed_id": prompt_idx,
                    "seed_prompt": seed,
                    "model_name": model_name,
                    "temperature": temp,
                    "generated_text": gen_result["generated_text"],
                    "full_text": gen_result["full_text"],
                    "metrics": quality_metrics,
                }
                results.append(entry)

                print(f"\n[{model_name} | Temp = {temp}]")
                print(f"  Continuation: {gen_result['generated_text']}")
                print(f"  Distinct-1: {quality_metrics['distinct_1']} | Distinct-2: {quality_metrics['distinct_2']} | Repetition-3: {quality_metrics['repetition_rate_3gram']}")

    os.makedirs("results", exist_ok=True)
    with open("results/temperature_generation_samples.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate a comprehensive analysis report
    generate_markdown_analysis(results, "results/temperature_analysis_report.md")
    print(f"\nSaved generation results to results/temperature_generation_samples.json and results/temperature_analysis_report.md")

def generate_markdown_analysis(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Qualitative Text Generation & Sampling Temperature Analysis\n\n")
        f.write("This report analyzes the autoregressive text generation output produced by Simple RNN, Bidirectional RNN, LSTM, and GRU models across low (0.2), moderate (0.7), and high (1.2) temperature values.\n\n")

        # Temperature summary table
        f.write("## 1. Temperature-Averaged Quality Metrics\n\n")
        f.write("| Model | Temperature | Avg Distinct-1 | Avg Distinct-2 | Avg 3-Gram Repetition | Avg Consecutive Repeat |\n")
        f.write("|---|---|---|---|---|---|\n")

        # Group by model and temp
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in results:
            key = (r["model_name"], r["temperature"])
            grouped[key].append(r["metrics"])

        for (m, t), metric_list in sorted(grouped.items()):
            avg_d1 = sum(x["distinct_1"] for x in metric_list) / len(metric_list)
            avg_d2 = sum(x["distinct_2"] for x in metric_list) / len(metric_list)
            avg_rep3 = sum(x["repetition_rate_3gram"] for x in metric_list) / len(metric_list)
            avg_consec = sum(x["consecutive_repeat_rate"] for x in metric_list) / len(metric_list)
            f.write(f"| **{m}** | {t} | {avg_d1:.3f} | {avg_d2:.3f} | {avg_rep3:.3f} | {avg_consec:.3f} |\n")

        f.write("\n## 2. Generated Text Samples per Seed Sequence\n\n")
        for prompt_idx, seed in enumerate(SEED_PROMPTS, 1):
            f.write(f"### Seed Sequence {prompt_idx}: *\"{seed}\"*\n\n")
            f.write("| Model | Temp | Generated Continuation | Distinct-1 | Repetition-3 |\n")
            f.write("|---|---|---|---|---|\n")
            for r in results:
                if r["seed_id"] == prompt_idx:
                    clean_text = r["generated_text"].replace("\n", " ").replace("|", "\\|")
                    f.write(f"| {r['model_name']} | {r['temperature']} | {clean_text} | {r['metrics']['distinct_1']} | {r['metrics']['repetition_rate_3gram']} |\n")
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=str, default="data/processed/vocab.json")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_gen_length", type=int, default=40)
    args = parser.parse_args()

    run_text_generation(
        vocab_path=args.vocab,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        max_gen_length=args.max_gen_length,
    )
