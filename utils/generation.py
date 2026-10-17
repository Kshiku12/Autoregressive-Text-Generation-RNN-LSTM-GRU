import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from .dataset import TextCorpusProcessor

def detokenize(tokens: List[str]) -> str:
    """
    Reconstructs natural human-readable text from tokens,
    formatting punctuation correctly without awkward spacing.
    """
    text = ""
    punctuation = {".", ",", "!", "?", ";", ":", ")", "]", "}", "'"}
    open_brackets = {"(", "[", "{"}

    for i, token in enumerate(tokens):
        if i == 0:
            text += token
        elif token in punctuation:
            text += token
        elif i > 0 and tokens[i - 1] in open_brackets:
            text += token
        elif token == '"':
            # Alternate quote handling or attach cleanly
            text += ' "'
        else:
            text += " " + token

    # Capitalize the start of sentences
    sentences = re_split_sentences(text)
    return sentences

def re_split_sentences(text: str) -> str:
    """Capitalizes the first character of sentences."""
    import re
    # Capitalize first character of text
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    # Capitalize after ., !, ?
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    return text

def generate_text_autoregressive(
    model: torch.nn.Module,
    processor: TextCorpusProcessor,
    seed_text: str,
    max_gen_length: int = 50,
    temperature: float = 1.0,
    seq_length: int = 20,
    device: torch.device = torch.device("cpu"),
    repetition_penalty: float = 1.0,
    ban_special_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Performs autoregressive sequential next-word generation from a seed sequence.
    
    Supports temperature scaling:
      - T < 0.5: Highly conservative, deterministic, high coherence, low diversity.
      - T ≈ 0.7 - 0.8: Balanced sweet spot.
      - T > 1.0: Creative, high entropy, higher diversity, potential degeneration.
    """
    model.eval()

    # Tokenize seed
    seed_tokens = processor.clean_and_tokenize(seed_text)
    if not seed_tokens:
        seed_tokens = ["the"]

    current_indices = processor.tokens_to_indices(seed_tokens)
    generated_indices: List[int] = []

    pad_idx = processor.word2idx.get(processor.PAD_TOKEN, 0)
    unk_idx = processor.word2idx.get(processor.UNK_TOKEN, 1)
    sos_idx = processor.word2idx.get(processor.SOS_TOKEN, 2)
    eos_idx = processor.word2idx.get(processor.EOS_TOKEN, 3)

    tokens_to_ban = [pad_idx, unk_idx, sos_idx] if ban_special_tokens else [pad_idx]

    with torch.no_grad():
        for _ in range(max_gen_length):
            # Extract context window of length seq_length
            if len(current_indices) < seq_length:
                # Left-pad with pad_idx
                padded_input = [pad_idx] * (seq_length - len(current_indices)) + current_indices
            else:
                padded_input = current_indices[-seq_length:]

            input_tensor = torch.tensor([padded_input], dtype=torch.long, device=device)

            logits, _ = model(input_tensor) # shape: (1, vocab_size)
            logits = logits[0] # shape: (vocab_size,)

            # Repetition penalty if desired
            if repetition_penalty > 1.0:
                for past_idx in set(current_indices[-seq_length:]):
                    if logits[past_idx] > 0:
                        logits[past_idx] /= repetition_penalty
                    else:
                        logits[past_idx] *= repetition_penalty

            # Mask banned special tokens
            for b_idx in tokens_to_ban:
                logits[b_idx] = -1e9

            # Temperature scaling
            if temperature <= 0.05:
                # Greedy selection
                next_token_idx = torch.argmax(logits).item()
            else:
                scaled_logits = logits / max(temperature, 1e-5)
                probs = F.softmax(scaled_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1).item()

            if next_token_idx == eos_idx:
                break

            generated_indices.append(next_token_idx)
            current_indices.append(next_token_idx)

    gen_tokens = processor.indices_to_tokens(generated_indices)
    full_tokens = seed_tokens + gen_tokens

    return {
        "model_name": getattr(model, "model_name", "Model"),
        "temperature": temperature,
        "seed_text": seed_text,
        "seed_tokens": seed_tokens,
        "generated_tokens": gen_tokens,
        "generated_text": detokenize(gen_tokens),
        "full_text": detokenize(full_tokens),
    }
