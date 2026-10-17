import math
from collections import Counter
from typing import List, Dict, Any, Tuple

def compute_perplexity(loss_val: float) -> float:
    """Computes perplexity = exp(cross_entropy_loss)."""
    try:
        return math.exp(min(loss_val, 100))
    except OverflowError:
        return float("inf")

def compute_top_k_accuracy(logits, targets, k: int = 5) -> float:
    """
    Computes top-k categorical accuracy.
    logits: torch.Tensor of shape (batch_size, vocab_size)
    targets: torch.Tensor of shape (batch_size)
    """
    import torch
    with torch.no_grad():
        _, pred_indices = logits.topk(k, dim=-1) # (batch_size, k)
        targets_expanded = targets.view(-1, 1).expand_as(pred_indices)
        correct = (pred_indices == targets_expanded).any(dim=-1).float().sum().item()
        return correct / targets.size(0)

def compute_distinct_ngrams(tokens: List[str], n: int = 1) -> float:
    """
    Computes Distinct-N diversity: ratio of unique n-grams to total n-grams.
    Higher values indicate greater lexical diversity and less collapse.
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

def compute_repetition_rate(tokens: List[str], n: int = 3) -> float:
    """
    Computes the proportion of repeated n-grams in the generated text.
    Lower values indicate less repetitive loops.
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)

def evaluate_generation_quality(generated_tokens: List[str], seed_tokens: List[str]) -> Dict[str, float]:
    """
    Evaluates diversity, repetition, and lexical features of generated tokens.
    """
    d1 = compute_distinct_ngrams(generated_tokens, n=1)
    d2 = compute_distinct_ngrams(generated_tokens, n=2)
    d3 = compute_distinct_ngrams(generated_tokens, n=3)
    rep2 = compute_repetition_rate(generated_tokens, n=2)
    rep3 = compute_repetition_rate(generated_tokens, n=3)

    # Consecutive repetition (e.g., 'the the the')
    consec_repeats = sum(1 for i in range(1, len(generated_tokens)) if generated_tokens[i] == generated_tokens[i-1])
    consec_rep_rate = consec_repeats / max(1, len(generated_tokens) - 1)

    return {
        "token_count": len(generated_tokens),
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "distinct_3": round(d3, 4),
        "repetition_rate_2gram": round(rep2, 4),
        "repetition_rate_3gram": round(rep3, 4),
        "consecutive_repeat_rate": round(consec_rep_rate, 4),
    }
