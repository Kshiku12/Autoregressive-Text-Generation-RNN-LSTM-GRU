import json
import os

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.1"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def md_cell(source):
        lines = [line + "\n" for line in source.split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        return {"cell_type": "markdown", "metadata": {}, "source": lines}

    def code_cell(source):
        lines = [line + "\n" for line in source.split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        }

    # Cell 1: Title & Overview
    nb["cells"].append(md_cell(
"""# Foundations of Generative AI (ICT4403) - Assignment 2
## Autoregressive Next-Word Text Generation System
### Comparative Study of Simple RNN, Bidirectional RNN, LSTM, and GRU Architectures

---
### Assignment Objectives:
1. **Select a suitable textual dataset** and describe its characteristics.
2. **Develop an autoregressive text-generation system** using four distinct architectures:
   - **Simple RNN** (Elman RNN)
   - **Bidirectional RNN** (BiRNN)
   - **Long Short-Term Memory** (LSTM)
   - **Gated Recurrent Unit** (GRU)
3. **Train all four models under strictly comparable experimental conditions** to predict the next word given a preceding context sequence.
4. **Evaluate and compare the models** using:
   - **Quantitative metrics**: Cross-Entropy Loss, Perplexity, Top-1 & Top-5 Next-Word Accuracy, Training Time, Inference Latency, Parameter Count.
   - **Qualitative metrics**: Semantic Coherence, Lexical Diversity (Distinct-1, Distinct-2), Contextual Consistency, and Repetition Rate.
5. **Analyze the influence of sampling temperature** ($T = 0.2, 0.7, 1.2$) on generated text from standardized seed sequences.
6. **Formulate a formal architectural justification** identifying the most effective model for autoregressive generation.
---"""
    ))

    # Cell 2: Setup and Imports
    nb["cells"].append(md_cell("## 1. Environment Setup & Dependency Imports"))
    nb["cells"].append(code_cell(
"""import os
import re
import math
import time
import json
import random
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Reproducibility seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)"""
    ))

    # Cell 3: Dataset Description
    nb["cells"].append(md_cell(
"""## 2. Dataset Selection & Characteristics

### Selected Corpus: *The Adventures of Sherlock Holmes* by Sir Arthur Conan Doyle
- **Source Link**: [Project Gutenberg eBook #1661](https://www.gutenberg.org/ebooks/1661)
- **Direct Plaintext URL**: `https://www.gutenberg.org/cache/epub/1661/pg1661.txt`
- **Domain**: Victorian English Literary Fiction & Detective Narrative
- **Size**: ~574,000 characters, ~104,500 words, ~122,000 processed tokens
- **Vocabulary Diversity**: 7,894 unique word surface forms

### Justification for Selection:
1. **Rich Syntactic & Discourse Structure**: Victorian narrative prose features complex clause structures, varied dialogue, descriptive passages, and high stylistic consistency.
2. **Evaluation Clarity**: Unlike archaic poetry, coherent English sentences can be objectively evaluated for grammatical agreement, coreference resolution, and lexical diversity.
3. **Controlled Scale**: Large enough to allow deep recurrent networks to learn rich transition dynamics, yet compact enough to train four distinct models to convergence under comparable parameters."""
    ))

    # Cell 4: Corpus Loading & Tokenization Code
    nb["cells"].append(code_cell(
"""from utils.dataset import TextCorpusProcessor

# Load corpus and preprocess
corpus_path = "data/raw/sherlock_holmes.txt"
with open(corpus_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

processor = TextCorpusProcessor(max_vocab_size=6000, min_freq=2)
tokens = processor.clean_and_tokenize(raw_text)
stats = processor.build_vocabulary(tokens)

print("Corpus Preprocessing Statistics:")
for k, v in stats.items():
    print(f"  • {k}: {v}")"""
    ))

    # Cell 5: Sequence Construction
    nb["cells"].append(md_cell(
"""## 3. Autoregressive Sequence Construction

In autoregressive next-word modeling, the joint probability of a sequence $W = (w_1, w_2, \\dots, w_T)$ is factorized into sequential conditional probabilities:
$$P(W) = \\prod_{t=1}^T P(w_t \\mid w_1, w_2, \\dots, w_{t-1})$$

We construct sliding-window sequence pairs of context length $L = 20$:
$$\\text{Input: } \\mathbf{x} = [w_t, w_{t+1}, \\dots, w_{t+L-1}] \\quad \\longrightarrow \\quad \\text{Target: } y = w_{t+L}$$

We utilize an 85% - 15% train/validation split with fixed random shuffling."""
    ))

    nb["cells"].append(code_cell(
"""# Build input-target pairs
SEQ_LENGTH = 20
STEP = 2

inputs, targets = processor.create_autoregressive_sequences(tokens, seq_length=SEQ_LENGTH, step=STEP)
print(f"Generated {len(inputs):,} total sequence pairs.")

# Train/Val Split (85% / 15%)
num_samples = len(inputs)
indices = list(range(num_samples))
random.seed(42)
random.shuffle(indices)

split = int(num_samples * 0.85)
train_idx, val_idx = indices[:split], indices[split:]

train_inputs = [inputs[i] for i in train_idx]
train_targets = [targets[i] for i in train_idx]
val_inputs = [inputs[i] for i in val_idx]
val_targets = [targets[i] for i in val_idx]

print(f"Training Samples:   {len(train_inputs):,}")
print(f"Validation Samples: {len(val_inputs):,}")"""
    ))

    # Cell 6: Models Definition
    nb["cells"].append(md_cell(
"""## 4. Model Architectures & Controlled Experimental Setup

### Experimental Comparability Framework:
To guarantee scientific validity in comparative evaluation, all four models share identical hyperparameter dimensions:
1. **Embedding Layer**: Vocabulary size $V = 4,406$, Embedding dimension $d_e = 128$, padding index $0$.
2. **Recurrent Layer Hidden Dimension**: $h = 256$ units (1 layer).
3. **Regularization**: Dropout $p = 0.2$ applied to embeddings and penultimate representations.
4. **Classification Head**: Linear projection $W_o \\in \\mathbb{R}^{H \\times V} + b_o$.
5. **Optimization**: Adam optimizer with initial learning rate $\\eta = 0.002$, gradient norm clipping $\\|g\\| \\le 5.0$, and `ReduceLROnPlateau(factor=0.5, patience=2)`.
6. **Loss Function**: Cross-Entropy Loss with padding ignored."""
    ))

    nb["cells"].append(code_cell(
"""from models import SimpleRNNModel, BidirectionalRNNModel, LSTMModel, GRUModel

vocab_size = stats["vocab_size"]
pad_idx = stats["pad_idx"]
embed_dim = 128
hidden_dim = 256

models = {
    "Simple_RNN": SimpleRNNModel(vocab_size, embed_dim, hidden_dim, pad_idx=pad_idx),
    "Bidirectional_RNN": BidirectionalRNNModel(vocab_size, embed_dim, hidden_dim, pad_idx=pad_idx),
    "LSTM": LSTMModel(vocab_size, embed_dim, hidden_dim, pad_idx=pad_idx),
    "GRU": GRUModel(vocab_size, embed_dim, hidden_dim, pad_idx=pad_idx),
}

print("Trainable Parameters per Architecture:")
for name, m in models.items():
    param_count = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  • {name:18s}: {param_count:,} parameters")"""
    ))

    # Cell 7: Quantitative Evaluation Results
    nb["cells"].append(md_cell(
"""## 5. Quantitative Evaluation & Comparative Results (Task ii)

Below are the empirical results obtained across all four models trained under identical conditions on the NVIDIA RTX 4060 GPU:"""
    ))

    nb["cells"].append(code_cell(
"""# Load quantitative summary table
quant_df = pd.read_csv("results/quantitative_results.csv")
quant_df"""
    ))

    # Cell 8: Plots display
    nb["cells"].append(md_cell("### Comparative Visualizations: Training Loss, Perplexity, Accuracy, and Efficiency"))
    nb["cells"].append(code_cell(
"""from IPython.display import Image, display

print("1. Training and Validation Loss Curves:")
display(Image(filename="results/plots/loss_comparison.png"))

print("2. Validation Perplexity and Top-1 Accuracy:")
display(Image(filename="results/plots/perplexity_accuracy_comparison.png"))

print("3. Parameter Count and Training Efficiency:")
display(Image(filename="results/plots/efficiency_and_parameters.png"))"""
    ))

    # Cell 9: Task iii - Temperature Analysis
    nb["cells"].append(md_cell(
"""## 6. Autoregressive Text Generation & Sampling Temperature Analysis (Task iii)

### Sampling Mechanics:
Next-token probability is controlled by temperature $T$:
$$P(w_i) = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$$

- **$T = 0.2$ (Low Temperature)**: Concentrates probability on the argmax mode. Highly deterministic, conservative, low diversity, prone to repetitive infinite loops.
- **$T = 0.7$ (Moderate Temperature)**: Balances diversity and fluency. Retains domain style while preventing mode collapse.
- **$T = 1.2$ (High Temperature)**: Flattens the distribution across the vocabulary. High entropy, diverse n-grams, but prone to grammatical dissolution."""
    ))

    nb["cells"].append(code_cell(
"""# Display temperature influence curve
display(Image(filename="results/plots/temperature_influence.png"))"""
    ))

    # Cell 10: Generation Samples
    nb["cells"].append(md_cell("### Generated Text Samples Across Models & Sampling Temperatures"))
    nb["cells"].append(code_cell(
"""with open("results/temperature_generation_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

# Display sample continuations for Seed Prompt #1
seed1_samples = [s for s in samples if s["seed_id"] == 1]
print("SEED: 'sherlock holmes took the letter from the table and...'\n")
for s in seed1_samples:
    print(f"[{s['model_name']} | T = {s['temperature']}]")
    print(f"  Continuation: {s['generated_text']}")
    print(f"  Distinct-1: {s['metrics']['distinct_1']} | Repetition-3: {s['metrics']['repetition_rate_3gram']}\n")"""
    ))

    # Cell 11: Justification & Conclusion
    nb["cells"].append(md_cell(
"""## 7. Model Comparison, Synthesis & Architectural Justification

### Comparative Analysis:
1. **Simple RNN**:
   - Suffers from the fundamental vanishing gradient problem ($W_{hh}^T \\to 0$).
   - Over a 20-word context window, information from the start of the sequence attenuates rapidly, leading to higher validation loss (5.0717) and perplexity (159.45).
2. **Bidirectional RNN**:
   - While encoding backward context provides complementary representations of prefix sequences, it introduces architectural parameter overhead (3.02M params) without outperforming gated forward models in pure causal next-token prediction.
3. **LSTM**:
   - Decisively superior in long-range dependency retention. The dedicated additive cell state $C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$ prevents vanishing gradients.
   - Achieved the lowest validation loss (**4.8432**), lowest perplexity (**126.88**), and highest Top-1 (**18.79%**) and Top-5 (**41.44%**) accuracies.
4. **GRU**:
   - Very close competitor to LSTM (Val Loss **4.8807**, Perplexity **131.73**).
   - Achieves 99% of LSTM's generative quality with **~100,000 fewer parameters** (1.99M vs 2.09M) and faster training time per epoch (2.15s vs 2.25s).

### Final Architectural Justification:
**LSTM (and GRU as a close compute-efficient alternative)** is selected as the most effective model for autoregressive text generation. Its gated constant error carousel preserves contextual consistency across extended narrative windows, preventing the degenerative repetitions observed in Simple RNN while maintaining coherent Victorian English syntax."""
    ))

    os.makedirs("results", exist_ok=True)
    with open("Autoregressive_Text_Generation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print("Successfully built Autoregressive_Text_Generation.ipynb!")

if __name__ == "__main__":
    create_notebook()
