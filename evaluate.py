import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_comparative_plots(history_path: str = "results/training_history.json", output_dir: str = "results/plots"):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(history_path):
        print(f"Error: {history_path} does not exist. Run train.py first.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    # Set aesthetic style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    colors = {
        "Simple_RNN": "#e74c3c",       # Vibrant Red
        "Bidirectional_RNN": "#e67e22",# Orange
        "LSTM": "#2980b9",             # Blue
        "GRU": "#27ae60",              # Green
    }
    model_labels = {
        "Simple_RNN": "Simple RNN",
        "Bidirectional_RNN": "Bidirectional RNN",
        "LSTM": "LSTM",
        "GRU": "GRU",
    }

    # 1. Training & Validation Loss Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, data in history.items():
        epochs = list(range(1, len(data["train_loss"]) + 1))
        c = colors.get(name, "#333333")
        label = model_labels.get(name, name)
        axes[0].plot(epochs, data["train_loss"], label=label, color=c, lw=2.2, marker="o", markersize=4)
        axes[1].plot(epochs, data["val_loss"], label=label, color=c, lw=2.2, marker="s", markersize=4)

    axes[0].set_title("Training Loss vs. Epochs", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].set_title("Validation Loss vs. Epochs", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    loss_path = os.path.join(output_dir, "loss_comparison.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved: {loss_path}")

    # 2. Perplexity and Accuracy Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, data in history.items():
        epochs = list(range(1, len(data["val_ppl"]) + 1))
        c = colors.get(name, "#333333")
        label = model_labels.get(name, name)
        axes[0].plot(epochs, data["val_ppl"], label=label, color=c, lw=2.2, marker="^", markersize=4)
        acc_pct = [acc * 100 for acc in data["val_acc1"]]
        axes[1].plot(epochs, acc_pct, label=label, color=c, lw=2.2, marker="d", markersize=4)

    axes[0].set_title("Validation Perplexity vs. Epochs", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Perplexity (Lower is Better)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].set_title("Validation Top-1 Next-Word Accuracy (%)", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="lower right", frameon=True)

    plt.tight_layout()
    ppl_acc_path = os.path.join(output_dir, "perplexity_accuracy_comparison.png")
    plt.savefig(ppl_acc_path, dpi=300)
    plt.close()
    print(f"Saved: {ppl_acc_path}")

    # 3. Bar Chart: Parameter Count & Training Efficiency
    summary_path = "results/quantitative_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

        models = [model_labels.get(s["Model"], s["Model"]) for s in summary]
        train_times = [s["Total Train Time (s)"] for s in summary]
        params = [s["Parameters"] / 1000 for s in summary]
        val_ppl = [s["Best Val Perplexity"] for s in summary]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        x = np.arange(len(models))
        bar_colors = [colors.get(s["Model"], "#333333") for s in summary]

        # Params
        axes[0].bar(x, params, color=bar_colors, width=0.55, edgecolor="black", alpha=0.85)
        axes[0].set_title("Model Parameters (k)", fontsize=12, fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=15, ha="right")
        for i, v in enumerate(params):
            axes[0].text(i, v + max(params)*0.01, f"{v:.1f}k", ha="center", fontsize=10, fontweight="bold")

        # Train Time
        axes[1].bar(x, train_times, color=bar_colors, width=0.55, edgecolor="black", alpha=0.85)
        axes[1].set_title("Total Training Time (s)", fontsize=12, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=15, ha="right")
        for i, v in enumerate(train_times):
            axes[1].text(i, v + max(train_times)*0.01, f"{v:.1f}s", ha="center", fontsize=10, fontweight="bold")

        # Perplexity
        axes[2].bar(x, val_ppl, color=bar_colors, width=0.55, edgecolor="black", alpha=0.85)
        axes[2].set_title("Best Validation Perplexity", fontsize=12, fontweight="bold")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=15, ha="right")
        for i, v in enumerate(val_ppl):
            axes[2].text(i, v + max(val_ppl)*0.01, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")

        plt.tight_layout()
        bar_path = os.path.join(output_dir, "efficiency_and_parameters.png")
        plt.savefig(bar_path, dpi=300)
        plt.close()
        print(f"Saved: {bar_path}")

    # 4. Temperature Impact Plot
    gen_samples_path = "results/temperature_generation_samples.json"
    if os.path.exists(gen_samples_path):
        with open(gen_samples_path, "r", encoding="utf-8") as f:
            gen_data = json.load(f)

        from collections import defaultdict
        temp_div = defaultdict(lambda: defaultdict(list))
        temp_rep = defaultdict(lambda: defaultdict(list))

        for entry in gen_data:
            m = entry["model_name"]
            t = entry["temperature"]
            temp_div[m][t].append(entry["metrics"]["distinct_1"])
            temp_rep[m][t].append(entry["metrics"]["repetition_rate_3gram"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for m, t_dict in temp_div.items():
            temps = sorted(t_dict.keys())
            mean_div = [np.mean(t_dict[t]) for t in temps]
            mean_rep = [np.mean(temp_rep[m][t]) for t in temps]
            c = colors.get(m, "#333333")
            lbl = model_labels.get(m, m)
            axes[0].plot(temps, mean_div, label=lbl, color=c, lw=2.2, marker="o")
            axes[1].plot(temps, mean_rep, label=lbl, color=c, lw=2.2, marker="s")

        axes[0].set_title("Lexical Diversity (Distinct-1) vs. Temperature", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Sampling Temperature (T)", fontsize=11)
        axes[0].set_ylabel("Distinct-1 Ratio (Higher = More Diverse)", fontsize=11)
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend(loc="lower right")

        axes[1].set_title("3-Gram Repetition Rate vs. Temperature", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Sampling Temperature (T)", fontsize=11)
        axes[1].set_ylabel("3-Gram Repetition (Lower = Less Repetitive)", fontsize=11)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        temp_plot_path = os.path.join(output_dir, "temperature_influence.png")
        plt.savefig(temp_plot_path, dpi=300)
        plt.close()
        print(f"Saved: {temp_plot_path}")

if __name__ == "__main__":
    generate_comparative_plots()
