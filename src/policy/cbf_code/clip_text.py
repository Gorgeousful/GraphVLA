"""Frozen OpenAI CLIP ViT-B/32 text encoding used by released CbF code."""

from __future__ import annotations

import gzip
import html
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import ftfy
import regex
import torch
from torch import Tensor, nn


@lru_cache()
def _bytes_to_unicode() -> dict[int, str]:
    values = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, characters), strict=True))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:]))


class _SimpleTokenizer:
    """OpenAI CLIP's byte-pair tokenizer, kept local to avoid a runtime package."""

    def __init__(self, bpe_path: str | Path) -> None:
        byte_encoder = _bytes_to_unicode()
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = [tuple(item.split()) for item in merges[1 : 49152 - 256 - 2 + 1]]
        vocabulary = list(byte_encoder.values())
        vocabulary += [token + "</w>" for token in byte_encoder.values()]
        vocabulary += ["".join(merge) for merge in merges]
        vocabulary += ["<|startoftext|>", "<|endoftext|>"]
        self.byte_encoder = byte_encoder
        self.encoder = dict(zip(vocabulary, range(len(vocabulary)), strict=True))
        self.ranks = dict(zip(merges, range(len(merges)), strict=True))
        self.cache = {token: token for token in vocabulary[-2:]}
        self.pattern = regex.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        while pairs:
            pair = min(pairs, key=lambda item: self.ranks.get(item, float("inf")))
            if pair not in self.ranks:
                break
            first, second = pair
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            pairs = _pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        text = regex.sub(r"\s+", " ", html.unescape(html.unescape(ftfy.fix_text(text)))).strip().lower()
        encoded = []
        for token in regex.findall(self.pattern, text):
            byte_token = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
            encoded.extend(self.encoder[piece] for piece in self._bpe(byte_token).split(" "))
        return encoded

    def tokenize(self, texts: Sequence[str], context_length: int = 77) -> Tensor:
        start, end = self.encoder["<|startoftext|>"], self.encoder["<|endoftext|>"]
        result = torch.zeros(len(texts), context_length, dtype=torch.long)
        for row, text in enumerate(texts):
            tokens = [start, *self.encode(text), end]
            if len(tokens) > context_length:
                raise RuntimeError(f"CLIP input is too long ({len(tokens)} tokens)")
            result[row, : len(tokens)] = torch.tensor(tokens)
        return result


class _QuickGELU(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value * torch.sigmoid(1.702 * value)


class _ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, attention_mask: Tensor) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict((
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", _QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width)),
        )))
        self.ln_2 = nn.LayerNorm(width)
        self.register_buffer("attention_mask", attention_mask, persistent=False)

    def forward(self, value: Tensor) -> Tensor:
        normalized = self.ln_1(value)
        attended = self.attn(
            normalized, normalized, normalized,
            need_weights=False,
            attn_mask=self.attention_mask.to(dtype=normalized.dtype, device=normalized.device),
        )[0]
        value = value + attended
        return value + self.mlp(self.ln_2(value))


class _Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attention_mask: Tensor) -> None:
        super().__init__()
        self.resblocks = nn.Sequential(*[
            _ResidualAttentionBlock(width, heads, attention_mask) for _ in range(layers)
        ])

    def forward(self, value: Tensor) -> Tensor:
        return self.resblocks(value)


class _ClipTextTower(nn.Module):
    """OpenAI CLIP text tower with parameter names matching the released checkpoint."""

    def __init__(self, *, context_length: int, vocab_size: int, width: int, layers: int, heads: int) -> None:
        super().__init__()
        mask = torch.empty(context_length, context_length).fill_(float("-inf")).triu_(1)
        self.transformer = _Transformer(width, layers, heads, mask)
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, width))

    def encode_text(self, text: Tensor) -> Tensor:
        value = self.token_embedding(text) + self.positional_embedding
        value = self.transformer(value.permute(1, 0, 2)).permute(1, 0, 2)
        value = self.ln_final(value)
        return value[torch.arange(value.shape[0], device=value.device), text.argmax(dim=-1)] @ self.text_projection


class FrozenClipTextEncoder:
    def __init__(self, model_path: str | Path, bpe_path: str | Path) -> None:
        scripted = torch.jit.load(str(model_path), map_location="cpu")
        state = scripted.state_dict()
        text_state = {
            key: value.float() for key, value in state.items()
            if key.startswith("transformer.") or key.startswith("token_embedding.")
            or key.startswith("ln_final.") or key in ("positional_embedding", "text_projection")
        }
        layers = len({key.split(".")[2] for key in text_state if key.startswith("transformer.resblocks.")})
        width = int(text_state["token_embedding.weight"].shape[1])
        self.model = _ClipTextTower(
            context_length=int(state["context_length"]),
            vocab_size=int(state["vocab_size"]),
            width=width,
            layers=layers,
            heads=width // 64,
        )
        self.model.load_state_dict(text_state, strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = _SimpleTokenizer(bpe_path)
        self.device = torch.device("cpu")
        self.cache: dict[str, Tensor] = {}
        self.lock = threading.Lock()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], *, device: torch.device) -> Tensor:
        device = torch.device(device)
        with self.lock:
            if self.device != device:
                self.model.to(device)
                self.device = device
                self.cache.clear()
            texts = [str(text) for text in texts]
            missing = [text for text in dict.fromkeys(texts) if text not in self.cache]
            if missing:
                with torch.autocast(device_type=device.type, enabled=False):
                    features = self.model.encode_text(
                        self.tokenizer.tokenize(missing).to(device)
                    ).float()
                    features = torch.nn.functional.normalize(features, p=2, dim=-1)
                self.cache.update({text: feature.cpu() for text, feature in zip(missing, features, strict=True)})
            return torch.stack([self.cache[text] for text in texts]).to(device)
