import os
import re
import json
from collections import Counter
from typing import List, Tuple, Dict, Any

class TextCorpusProcessor:
    """
    Handles text cleaning, tokenization, vocabulary creation,
    and sequence construction for autoregressive next-word language modeling.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, max_vocab_size: int = 6000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.stats: Dict[str, Any] = {}

    @staticmethod
    def clean_and_tokenize(raw_text: str) -> List[str]:
        """
        Cleans text and extracts tokens, preserving punctuation as distinct tokens
        so the generated output retains natural grammatical structure.
        """
        # Normalize whitespace, quotes, and hyphens
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        text = re.sub(r"--+", " - ", text)

        # Separate punctuation with whitespace while preserving words and contractions
        # e.g., "don't" stays "don't", but "hello." becomes "hello ."
        # Matches word contractions, numbers, individual words, or individual punctuation marks
        token_pattern = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[.,!?;:\"()\-]")
        raw_tokens = token_pattern.findall(text)

        # Convert to lowercase tokens
        tokens = [t.lower() for t in raw_tokens if t.strip()]
        return tokens

    def build_vocabulary(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Constructs vocabulary index mappings with special tokens and frequency pruning.
        """
        counter = Counter(tokens)
        total_tokens = len(tokens)
        unique_tokens = len(counter)

        # Reserve indices for special tokens
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.idx2word = {idx: token for idx, token in enumerate(special_tokens)}

        # Add most common words that meet min_freq requirement
        available_slots = self.max_vocab_size - len(special_tokens)
        most_common = [
            word for word, count in counter.most_common()
            if count >= self.min_freq
        ][:available_slots]

        for word in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)

        # Calculate coverage
        covered_count = sum(counter[word] for word in self.word2idx if word in counter)
        coverage_pct = (covered_count / total_tokens) * 100 if total_tokens > 0 else 0.0

        self.stats = {
            "total_tokens": total_tokens,
            "unique_tokens_raw": unique_tokens,
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "coverage_percentage": round(coverage_pct, 2),
            "pad_idx": self.word2idx[self.PAD_TOKEN],
            "unk_idx": self.word2idx[self.UNK_TOKEN],
            "sos_idx": self.word2idx[self.SOS_TOKEN],
            "eos_idx": self.word2idx[self.EOS_TOKEN],
        }

        return self.stats

    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """Maps a sequence of word tokens to vocabulary indices."""
        unk_idx = self.word2idx[self.UNK_TOKEN]
        return [self.word2idx.get(t, unk_idx) for t in tokens]

    def indices_to_tokens(self, indices: List[int]) -> List[str]:
        """Maps a sequence of vocabulary indices back to words."""
        return [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]

    def create_autoregressive_sequences(
        self, tokens: List[str], seq_length: int = 20, step: int = 1
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Creates sliding window (input_seq, target_word) pairs for next-word prediction.
        Input: [w_t, w_{t+1}, ..., w_{t+seq_len-1}]
        Target: w_{t+seq_len}
        """
        token_indices = self.tokens_to_indices(tokens)
        input_sequences = []
        target_words = []

        for i in range(0, len(token_indices) - seq_length, step):
            seq_in = token_indices[i : i + seq_length]
            target = token_indices[i + seq_length]
            input_sequences.append(seq_in)
            target_words.append(target)

        return input_sequences, target_words

    def save_vocab(self, filepath: str):
        """Saves vocabulary and configuration to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "word2idx": self.word2idx,
            "stats": self.stats,
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_vocab(self, filepath: str):
        """Loads vocabulary and configuration from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.stats = data.get("stats", {})
        self.max_vocab_size = data.get("max_vocab_size", len(self.word2idx))
        self.min_freq = data.get("min_freq", 2)
        self.vocab_size = len(self.word2idx)
