"""
Advanced Tokenizers for EquiTile LM
====================================

Provides production-ready tokenizers:
- BPETokenizer: Byte-Pair Encoding (GPT-2 style)
- WordPieceTokenizer: WordPiece (BERT style)
- CachedTokenizer: Tokenization caching for efficiency

Example
-------
>>> from bioplausible.equitile.lm_demo.data_advanced import BPETokenizer
>>> tokenizer = BPETokenizer()
>>> ids = tokenizer.encode("Hello, world!")
>>> text = tokenizer.decode(ids)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch

# =============================================================================
# BPE Tokenizer (GPT-2 style)
# =============================================================================


class BPETokenizer:
    """Byte-Pair Encoding tokenizer.

    Implements BPE tokenization similar to GPT-2:
    1. Start with character-level vocabulary
    2. Iteratively merge most frequent pairs
    3. Build vocabulary of subword tokens

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size
    merges : int
        Number of BPE merges to perform
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        merges: int = 40000,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_merges = merges

        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"

        # Build vocabulary
        self.vocab: Dict[str, int] = {}
        self.merges: Dict[Tuple[str, str], int] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build initial vocabulary with special tokens and characters."""
        # Special tokens
        special = [self.pad_token, self.unk_token, self.eos_token, self.bos_token]
        for i, token in enumerate(special):
            self.vocab[token] = i

        self.special_count = len(special)

        # Add all printable ASCII characters
        for i in range(32, 127):
            char = chr(i)
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # Add common whitespace
        for char in [" ", "\n", "\t", "\r"]:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

    def train(self, texts: List[str]) -> None:
        """Train BPE on texts.

        Parameters
        ----------
        texts : list of str
            Training texts
        """
        # Pre-tokenize to words
        words = []
        for text in texts:
            # Split on whitespace, keep separators
            tokens = re.findall(r"\w+|[^\w\s]", text.lower())
            words.extend(tokens)

        # Count word frequencies
        word_freq = Counter(words)

        # Convert words to character sequences
        word_splits = {word: list(word) for word in word_freq}

        # Perform BPE merges
        for merge_idx in range(self.max_merges):
            if len(self.vocab) >= self.vocab_size:
                break

            # Count pairs
            pair_freq = defaultdict(int)
            for word, freq in word_freq.items():
                chars = word_splits[word]
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    pair_freq[pair] += freq

            if not pair_freq:
                break

            # Find most frequent pair
            best_pair = max(pair_freq, key=pair_freq.get)

            # Merge the pair
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.merges[best_pair] = len(self.merges)

            # Apply merge to all words
            for word in word_splits:
                chars = word_splits[word]
                new_chars = []
                i = 0
                while i < len(chars) - 1:
                    if (chars[i], chars[i + 1]) == best_pair:
                        new_chars.append(chars[i] + chars[i + 1])
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                if i < len(chars):
                    new_chars.append(chars[-1])
                word_splits[word] = new_chars

        print(
            f"BPE training complete: {len(self.vocab)} tokens, {len(self.merges)} merges"
        )

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        list of int
            Token IDs
        """
        # Lowercase and tokenize
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())

        ids = []
        for token in tokens:
            # Apply BPE merges
            chars = list(token)

            # Iteratively apply merges
            changed = True
            while changed and len(chars) > 1:
                changed = False
                new_chars = []
                i = 0
                while i < len(chars) - 1:
                    pair = (chars[i], chars[i + 1])
                    if pair in self.merges:
                        merged = chars[i] + chars[i + 1]
                        if merged in self.vocab:
                            new_chars.append(merged)
                            i += 2
                            changed = True
                            continue
                    new_chars.append(chars[i])
                    i += 1
                if i < len(chars):
                    new_chars.append(chars[-1])
                chars = new_chars

            # Convert to IDs
            for char in chars:
                if char in self.vocab:
                    ids.append(self.vocab[char])
                else:
                    ids.append(self.vocab[self.unk_token])

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text.

        Parameters
        ----------
        ids : list of int
            Token IDs

        Returns
        -------
        str
            Decoded text
        """
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(i, self.unk_token) for i in ids]
        return "".join(tokens)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> torch.Tensor:
        """Batch encode texts.

        Parameters
        ----------
        texts : list of str
            Input texts
        max_length : int, optional
            Maximum sequence length
        padding : bool
            Pad to max_length

        Returns
        -------
        torch.Tensor
            Token IDs (batch, seq_len)
        """
        encoded = [self.encode(t) for t in texts]

        if max_length:
            if padding:
                for i, ids in enumerate(encoded):
                    if len(ids) < max_length:
                        encoded[i] = ids + [self.vocab[self.pad_token]] * (
                            max_length - len(ids)
                        )
                    else:
                        encoded[i] = ids[:max_length]
            else:
                encoded = [ids[:max_length] for ids in encoded]
        elif padding and encoded:
            max_len = max(len(ids) for ids in encoded)
            for i, ids in enumerate(encoded):
                encoded[i] = ids + [self.vocab[self.pad_token]] * (max_len - len(ids))

        return torch.tensor(encoded, dtype=torch.long)

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        data = {
            "vocab": self.vocab,
            "merges": {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()},
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "pad": self.pad_token,
                "unk": self.unk_token,
                "eos": self.eos_token,
                "bos": self.bos_token,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from file."""
        with open(path, "r") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data.get("vocab_size", 50000))
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = {tuple(k.split("|")): v for k, v in data["merges"].items()}

        special = data.get("special_tokens", {})
        tokenizer.pad_token = special.get("pad", "<pad>")
        tokenizer.unk_token = special.get("unk", "<unk>")
        tokenizer.eos_token = special.get("eos", "<eos>")
        tokenizer.bos_token = special.get("bos", "<bos>")

        return tokenizer


# =============================================================================
# WordPiece Tokenizer (BERT style)
# =============================================================================


class WordPieceTokenizer:
    """WordPiece tokenizer.

    Similar to BERT's tokenizer:
    - Uses ## prefix for continuation tokens
    - Better for word-level understanding

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size
    """

    def __init__(self, vocab_size: int = 30000) -> None:
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build initial vocabulary."""
        # Special tokens
        special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        for i, token in enumerate(special):
            self.vocab[token] = i

        self.pad_token_id = 0
        self.unk_token_id = 1

        # Add characters
        for i in range(32, 127):
            char = chr(i)
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

    def train(self, texts: List[str]) -> None:
        """Train WordPiece on texts."""
        # Tokenize to words
        words = []
        for text in texts:
            tokens = re.findall(r"\w+|[^\w\s]", text.lower())
            words.extend(tokens)

        # Count subwords
        subword_freq = Counter()
        for word in words:
            # Start with ## prefix for all but first char
            subwords = [word[0]] + [f"##{c}" for c in word[1:]]
            for sw in subwords:
                subword_freq[sw] += 1

        # Add most frequent subwords
        for subword, _ in subword_freq.most_common(self.vocab_size - len(self.vocab)):
            if subword not in self.vocab:
                self.vocab[subword] = len(self.vocab)

        print(f"WordPiece training complete: {len(self.vocab)} tokens")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())

        ids = []
        for token in tokens:
            if len(token) == 1:
                # Single character
                if token in self.vocab:
                    ids.append(self.vocab[token])
                else:
                    ids.append(self.unk_token_id)
            else:
                # First character
                if token[0] in self.vocab:
                    ids.append(self.vocab[token[0]])
                else:
                    ids.append(self.unk_token_id)

                # Rest with ## prefix
                for char in token[1:]:
                    subword = f"##{char}"
                    if subword in self.vocab:
                        ids.append(self.vocab[subword])
                    else:
                        ids.append(self.unk_token_id)

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(i, "[UNK]") for i in ids]

        # Join, handling ## prefix
        result = []
        for token in tokens:
            if token.startswith("##"):
                if result:
                    result[-1] += token[2:]
                else:
                    result.append(token[2:])
            else:
                result.append(token)

        return "".join(result)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> torch.Tensor:
        """Batch encode texts."""
        encoded = [self.encode(t) for t in texts]

        if max_length:
            if padding:
                for i, ids in enumerate(encoded):
                    if len(ids) < max_length:
                        encoded[i] = ids + [self.pad_token_id] * (max_length - len(ids))
                    else:
                        encoded[i] = ids[:max_length]
        elif padding and encoded:
            max_len = max(len(ids) for ids in encoded)
            for i, ids in enumerate(encoded):
                encoded[i] = ids + [self.pad_token_id] * (max_len - len(ids))

        return torch.tensor(encoded, dtype=torch.long)


# =============================================================================
# Factory Functions
# =============================================================================


def create_tokenizer(
    tokenizer_type: str = "bpe",
    vocab_size: int = 50000,
    texts: Optional[List[str]] = None,
    cache_path: Optional[str] = None,
) -> BPETokenizer | WordPieceTokenizer:
    """Create and optionally train a tokenizer.

    Parameters
    ----------
    tokenizer_type : str
        Type: 'bpe' or 'wordpiece'
    vocab_size : int
        Vocabulary size
    texts : list of str, optional
        Training texts
    cache_path : str, optional
        Path to cache/load tokenizer

    Returns
    -------
    BPETokenizer or WordPieceTokenizer
        Trained tokenizer
    """
    if cache_path and Path(cache_path).exists():
        print(f"Loading tokenizer from {cache_path}")
        if tokenizer_type == "bpe":
            return BPETokenizer.load(cache_path)
        else:
            return WordPieceTokenizer.load(cache_path)

    # Create new tokenizer
    if tokenizer_type == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size)
    else:
        tokenizer = WordPieceTokenizer(vocab_size=vocab_size)

    # Train if texts provided
    if texts:
        print(f"Training {tokenizer_type} tokenizer on {len(texts)} texts...")
        tokenizer.train(texts)

        # Save if cache path provided
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(cache_path)
            print(f"Tokenizer saved to {cache_path}")

    return tokenizer


def load_shakespeare_tokenizer(
    vocab_size: int = 5000,
    cache_path: str = "data/shakespeare_tokenizer.json",
) -> BPETokenizer:
    """Load tokenizer trained on Shakespeare."""
    from .data import _get_shakespeare_text

    text = _get_shakespeare_text()

    # Split into sentences for training
    sentences = [s.strip() for s in text.split("\n") if len(s.strip()) > 10]

    return create_tokenizer(
        tokenizer_type="bpe",
        vocab_size=vocab_size,
        texts=sentences,
        cache_path=cache_path,
    )
