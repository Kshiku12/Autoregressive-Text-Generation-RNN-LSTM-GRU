# Autoregressive Text Generation System
## Comparative Evaluation of Simple RNN, Bidirectional RNN, LSTM, and GRU Architectures

**Course**: Foundations of Generative AI (ICT4403) — Assignment 2  
**Implementation**: PyTorch with NVIDIA CUDA GPU Acceleration  
**Corpus**: *The Adventures of Sherlock Holmes* by Sir Arthur Conan Doyle (Project Gutenberg eBook #1661)

---

## Table of Contents
1. [Problem Statement & Objective](#1-problem-statement--objective)
2. [Dataset Description & Source Link](#2-dataset-description--source-link)
3. [Brief Description of Implemented Models](#3-brief-description-of-implemented-models)
4. [Execution Steps & Required Dependencies](#4-execution-steps--required-dependencies)
5. [Experimental Results Obtained](#5-experimental-results-obtained)
6. [Model Comparison & Evaluation](#6-model-comparison--evaluation)
7. [Sampling-Temperature Analysis](#7-sampling-temperature-analysis)
8. [Conclusion & Best-Performing Model Selection](#8-conclusion--best-performing-model-selection)
9. [Repository Organization & Submission Procedure](#9-repository-organization--submission-procedure)

---

## 1. Problem Statement & Objective

Autoregressive language modeling is the foundational paradigm underpinning modern generative sequence models. In an autoregressive formulation, given an observed sequence of tokens $W = (w_1, w_2, \dots, w_T)$, the joint probability is factorized into the product of conditional next-token probabilities:

$$P(W) = \prod_{t=1}^T P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

The primary objectives of this project are:
1. **Develop an end-to-end autoregressive text-generation system** implementing four distinct recurrent architectures: **Simple RNN**, **Bidirectional RNN**, **LSTM**, and **GRU**.
2. **Train all four models under strictly comparable, controlled experimental conditions** (identical vocabulary, embedding dimension, hidden dimensions, context length, learning rate, and loss function) to predict the next word from a preceding sequence.
3. **Quantitatively evaluate and compare the models** using Cross-Entropy Loss, Perplexity ($PPL = e^{\mathcal{L}}$), Top-1 and Top-5 word prediction accuracy, training time, parameter count, and per-token inference latency.
4. **Qualitatively evaluate generated text** across coherence, lexical diversity (Distinct-1, Distinct-2), contextual consistency, and repetition rates.
5. **Analyze the impact of sampling temperature** ($T \in \{0.2, 0.7, 1.2\}$) on generative behavior and justify which model provides the most effective autoregressive generation for the selected corpus.

---

## 2. Dataset Description & Source Link

- **Title**: *The Adventures of Sherlock Holmes*
- **Author**: Sir Arthur Conan Doyle
- **Source Link**: [Project Gutenberg eBook #1661](https://www.gutenberg.org/ebooks/1661)
- **Direct Plaintext URL**: `https://www.gutenberg.org/cache/epub/1661/pg1661.txt`
- **Domain**: Victorian English Literary Fiction & Detective Narrative

### Characteristics & Preprocessing Pipeline:
1. **Header & Footer Stripping**: Project Gutenberg legal disclaimers and metadata are stripped using regular expression delimiters to retain pure literary prose.
2. **Tokenization with Punctuation Preservation**: Tokens are extracted such that grammatical punctuation (`.`, `,`, `!`, `?`, `;`, `:`, `"`, `'`) is retained as discrete tokens. This ensures the generative system learns sentence boundaries and punctuation syntax rather than generating unpunctuated text.
3. **Vocabulary Construction**:
   - Special reserved tokens: `<PAD>` (index 0), `<UNK>` (index 1), `<SOS>` (index 2), `<EOS>` (index 3).
   - Frequency thresholding: Words with frequency $\ge 2$ are indexed up to maximum vocabulary size.
   - **Total Tokens**: 121,942
   - **Unique Surface Forms**: 7,894
   - **Vocabulary Size**: 4,406 tokens
   - **Token Coverage**: **97.14%** (infrequent rare words mapped to `<UNK>`).
4. **Sequence Construction**:
   - Sliding context window of $L = 20$ words with stride $S = 2$.
   - **Total Sequence Pairs**: 60,961 pairs of $([w_t, \dots, w_{t+19}], w_{t+20})$.
   - **Train Split (85%)**: 51,816 samples.
   - **Validation Split (15%)**: 9,145 samples.

---

## 3. Brief Description of Implemented Models

All models are implemented in PyTorch using a unified classification interface:

```
Input Sequence (B, 20) -> Embedding (B, 20, 128) -> Dropout(0.2) -> Recurrent Layer -> Penultimate Hidden -> FC Classifier -> Logits (B, 4406)
```

### 1. Simple RNN (Elman RNN)
- **Recurrence**: $h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$
- **Parameters**: 1,795,126
- **Characteristics**: Single continuous hidden state. Prone to vanishing and exploding gradients when backpropagating through time (BPTT) over 20-step horizons, limiting its ability to maintain long-range dependencies.

### 2. Bidirectional RNN (BiRNN)
- **Recurrence**: Encodes the prefix sequence bidirectionally: forward $h_t^f$ from $w_1 \to w_T$ and backward $h_t^b$ from $w_T \to w_1$. The final prefix representation concatenates the forward terminal state and backward terminal state: $[h_T^f \,;\, h_1^b] \in \mathbb{R}^{2H}$.
- **Parameters**: 3,021,878
- **Characteristics**: Evaluates whether backward representation of the prefix context enriches next-token prediction without leaking future tokens.

### 3. Long Short-Term Memory (LSTM)
- **Recurrence**: Uses three multiplicative gates (input $i_t$, forget $f_t$, output $o_t$) and an additive cell state $C_t$:
  $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
  $$\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c), \quad C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
  $$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(C_t)$$
- **Parameters**: 2,091,574
- **Characteristics**: The Constant Error Carousel (CEC) along $C_t$ enables gradients to flow across the 20-word horizon without exponential decay, allowing long-term syntactic retention.

### 4. Gated Recurrent Unit (GRU)
- **Recurrence**: Combines cell state and hidden state using two gates (reset $r_t$ and update $z_t$):
  $$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r), \quad z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$
  $$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t] + b), \quad h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
- **Parameters**: 1,992,758
- **Characteristics**: Computationally leaner than LSTM (2 gates vs. 3 gates, ~100k fewer parameters) while maintaining similar protective gradient pathways.

---

## 4. Execution Steps & Required Dependencies

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- CUDA-compatible GPU (e.g., NVIDIA GeForce RTX 4060 Laptop GPU) or CPU.

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Kshiku12/Autoregressive-Text-Generation-RNN-LSTM-GRU.git
cd Autoregressive-Text-Generation-RNN-LSTM-GRU
pip install -r requirements.txt
```

### Reproducing the Complete Pipeline
```bash
# Step 1: Download and clean the corpus
python scripts/download_data.py

# Step 2: Train all 4 models under identical conditions (15 epochs)
python train.py --epochs 15 --batch_size 128 --seq_length 20 --step 2

# Step 3: Run autoregressive generation across temperatures (T = 0.2, 0.7, 1.2)
python generate.py

# Step 4: Generate evaluation plots and charts
python evaluate.py

# Step 5: Generate submission Word (.docx) and PDF (.pdf) documents
python generate_submission_doc.py "Kumar Kshitij" "230953488" "https://github.com/Kshiku12/Autoregressive-Text-Generation-RNN-LSTM-GRU"
```

---

## 5. Experimental Results Obtained

### Strictly Controlled Hyperparameters:
| Hyperparameter | Value |
|---|---|
| Embedding Dimension | 128 |
| Hidden Dimension | 256 |
| Sequence Length ($L$) | 20 words |
| Batch Size | 128 |
| Optimization Algorithm | Adam ($\beta_1=0.9, \beta_2=0.999$) |
| Initial Learning Rate | $2 \times 10^{-3}$ (with ReduceLROnPlateau) |
| Gradient Clipping | $\|g\| \le 5.0$ |
| Regularization | Dropout $p = 0.2$ |
| Number of Epochs | 15 |
| Hardware Platform | NVIDIA GeForce RTX 4060 Laptop GPU |

### Quantitative Comparison Table:
| Model | Trainable Parameters | Best Val Loss | Best Val Perplexity | Best Val Top-1 Acc (%) | Best Val Top-5 Acc (%) | Avg Epoch Time (s) | Total Train Time (s) | Inference Latency (ms/token) |
|---|---|---|---|---|---|---|---|---|
| **Simple RNN** | 1,795,126 | 5.0717 | 159.45 | 18.13% | 39.08% | 1.67s | 25.06s | **0.167 ms** |
| **Bidirectional RNN** | 3,021,878 | 5.2430 | 189.23 | 16.51% | 36.58% | 2.44s | 36.66s | 0.224 ms |
| **LSTM** | 2,091,574 | **4.8432** | **126.88** | **18.79%** | **41.44%** | 2.25s | 33.74s | 0.630 ms |
| **GRU** | **1,992,758** | 4.8807 | 131.73 | 18.12% | 40.12% | 2.15s | 32.33s | 0.554 ms |

---

## 6. Model Comparison & Evaluation

![Training and Validation Loss](results/plots/loss_comparison.png)
*Figure 1: Training and Validation Loss across 15 epochs for all four recurrent architectures.*

![Perplexity and Top-1 Accuracy](results/plots/perplexity_accuracy_comparison.png)
*Figure 2: Validation Perplexity (lower is better) and Top-1 Next-Word Accuracy (%) curves.*

![Model Parameters and Efficiency](results/plots/efficiency_and_parameters.png)
*Figure 3: Trainable parameters, total training duration, and best validation perplexity.*

### In-Depth Analysis:
1. **Vanishing Gradients in Simple RNN**: While Simple RNN trains fast (1.67s/epoch), its validation perplexity flattens early at **159.45**. The lack of additive memory pathways prevents it from retaining multi-word contextual dependencies across the 20-word context window.
2. **Bidirectional RNN Limitations in Causal Generation**: BiRNN has the largest parameter count (3.02M) due to its dual recurrent streams. However, in autoregressive next-word prediction, encoding the prefix in reverse order provides diminishing returns compared to forward-gated memory, resulting in higher validation loss (5.2430) and perplexity (189.23).
3. **LSTM Superiority**: LSTM achieved the lowest validation loss (**4.8432**), lowest perplexity (**126.88**), and highest Top-1 (**18.79%**) and Top-5 (**41.44%**) accuracies. The forget and input gates successfully isolate irrelevant past tokens while preserving key narrative entities.
4. **GRU Efficiency**: GRU achieved performance remarkably close to LSTM (Val Loss 4.8807 vs. 4.8432, Perplexity 131.73 vs. 126.88) while operating with **98,816 fewer parameters** (1.99M vs. 2.09M) and training ~4% faster per epoch.

---

## 7. Sampling-Temperature Analysis

Autoregressive sequential generation was conducted across three sampling temperatures:
- **$T = 0.2$** (Conservative / Low Entropy)
- **$T = 0.7$** (Balanced / Moderate Entropy)
- **$T = 1.2$** (Diverse / High Entropy)

![Temperature Influence](results/plots/temperature_influence.png)
*Figure 4: Influence of temperature on Lexical Diversity (Distinct-1) and 3-Gram Repetition Rate.*

### Aggregate Metric Comparison Across Temperatures:
| Model | Temperature | Avg Distinct-1 (Diversity) | Avg Distinct-2 | Avg 3-Gram Repetition Rate |
|---|---|---|---|---|
| **Simple RNN** | $T = 0.2$ | 0.542 | 0.731 | 0.165 |
| **Simple RNN** | $T = 0.7$ | 0.768 | 0.949 | 0.026 |
| **Simple RNN** | $T = 1.2$ | **0.842** | **0.994** | **0.000** |
| **Bidirectional RNN** | $T = 0.2$ | 0.448 | 0.654 | 0.263 |
| **Bidirectional RNN** | $T = 0.7$ | 0.695 | 0.923 | 0.026 |
| **Bidirectional RNN** | $T = 1.2$ | **0.865** | **0.994** | **0.000** |
| **LSTM** | $T = 0.2$ | 0.531 | 0.705 | 0.211 |
| **LSTM** | $T = 0.7$ | 0.756 | 0.942 | 0.013 |
| **LSTM** | $T = 1.2$ | **0.875** | **1.000** | **0.000** |
| **GRU** | $T = 0.2$ | 0.519 | 0.718 | 0.211 |
| **GRU** | $T = 0.7$ | 0.725 | 0.904 | 0.026 |
| **GRU** | $T = 1.2$ | **0.862** | **1.000** | **0.000** |

### Qualitative Analysis of Seed Sequence Generations:

#### Prompt: *"sherlock holmes took the letter from the table and..."*
- **Low Temperature ($T = 0.2$)**:
  - *LSTM*: `"...said that he had been in the house, and that he was a very small of the house, and that he had been in the room, and that he had been in the room..."`
  - *Analysis*: Strong mode collapse. Probability concentrates in the top 1-2 words ("in the room", "of the house"), creating degenerate infinite loops. 3-gram repetition exceeds 21%.
- **Balanced Temperature ($T = 0.7$)**:
  - *LSTM*: `"...looked over the other, and his head was a few words which had been drawn down the stairs. The morning of the window was a little light upon the stairs, and he was able to see..."`
  - *Analysis*: Excellent coherence, natural syntactic cadence, vivid detective imagery, and zero 3-gram repetition. Distinct-1 reaches **0.756**.
- **High Temperature ($T = 1.2$)**:
  - *LSTM*: `"...seemed rapidly by a light was indeed very little, however. You are too good a fellow, for the 85 of your letter. I had just to my niece, and what had been taken..."`
  - *Analysis*: Highest lexical diversity (Distinct-1: **0.875**), but semantic focus wanders into unrelated fragments.

---

## 8. Conclusion & Best-Performing Model Selection

### Architectural Winner: **Long Short-Term Memory (LSTM)**
1. **Quantitative Dominance**: LSTM achieved the lowest validation loss (**4.8432**), lowest validation perplexity (**126.88**), and highest Top-1 (**18.79%**) and Top-5 (**41.44%**) next-word prediction accuracies.
2. **Qualitative Superiority**: At the optimal sampling temperature of $T = 0.7$, LSTM produced the most syntactically coherent, contextually appropriate text adhering to Arthur Conan Doyle's narrative style.
3. **Runner-Up Distinction (GRU)**: GRU serves as the most parameter-efficient model, achieving 99% of LSTM's perplexity performance with **98,816 fewer parameters** and faster training throughput.

---

## 9. Repository Organization & Submission Procedure

### Project Directory Structure:
```
DL project/
├── data/
│   ├── raw/
│   │   └── sherlock_holmes.txt              # Gutenberg eBook #1661 plaintext
│   └── processed/
│       └── vocab.json                       # Vocabulary and token mappings
├── models/
│   ├── __init__.py
│   ├── simple_rnn.py                        # Simple Elman RNN
│   ├── bidirectional_rnn.py                 # Bidirectional RNN
│   ├── lstm.py                              # Long Short-Term Memory
│   └── gru.py                               # Gated Recurrent Unit
├── utils/
│   ├── __init__.py
│   ├── dataset.py                           # Tokenizer, vocab builder, sequence loader
│   ├── metrics.py                           # Perplexity, Top-k accuracy, Distinct-N
│   └── generation.py                        # Autoregressive sampling with temperature
├── checkpoints/
│   ├── Simple_RNN_best.pt                   # Best model checkpoint weights
│   ├── Bidirectional_RNN_best.pt
│   ├── LSTM_best.pt
│   └── GRU_best.pt
├── results/
│   ├── plots/
│   │   ├── loss_comparison.png              # Train/Val loss curves
│   │   ├── perplexity_accuracy_comparison.png # Perplexity and accuracy curves
│   │   ├── efficiency_and_parameters.png    # Training speed and param charts
│   │   └── temperature_influence.png        # Distinct-N and repetition vs Temp
│   ├── quantitative_results.csv             # Full benchmark table (CSV)
│   ├── quantitative_summary.json            # Full benchmark metrics (JSON)
│   ├── training_history.json                # Epoch-by-epoch loss and accuracy logs
│   ├── temperature_generation_samples.json  # Generated samples across models/temps
│   └── temperature_analysis_report.md       # Comprehensive markdown generation report
├── scripts/
│   ├── download_data.py                     # Data download & cleaning script
│   └── build_notebook.py                    # Script to construct .ipynb notebook
├── Autoregressive_Text_Generation.ipynb     # Self-contained Jupyter Notebook
├── train.py                                 # End-to-end multi-model training script
├── evaluate.py                              # Plot generation and evaluation script
├── generate.py                              # Generation across seeds and temperatures
├── generate_submission_doc.py               # Step 3 Word (.docx) & Step 4 (.pdf) generator
├── requirements.txt                         # Python package dependencies
├── KumarKshitij_230953488_FGAI_Assignment2.docx    # Word document (Step 3)
├── KumarKshitij_230953488_FGAI_Assignment2.pdf     # Converted PDF document (Step 4 & 5)
└── README.md                                # Comprehensive project documentation
```

### Submission Checklist for LMS (Steps 1–5):
- [x] **Step 1**: All four models implemented, trained, evaluated, and analyzed under identical conditions.
- [x] **Step 2**: Git repository ready with `Autoregressive_Text_Generation.ipynb`, Python source files, plots, results, and comprehensive `README.md`.
- [x] **Step 3**: Created Word document `YourName_RegNum_FGAI_Assignment2.docx` with student name, registration number, assignment title, system title, dataset link, and GitHub repository link.
- [x] **Step 4**: Converted Word document to `YourName_RegNum_FGAI_Assignment2.pdf`.
- [x] **Step 5**: Upload `YourName_RegNum_FGAI_Assignment2.pdf` to the LMS assignment portal before the deadline.
