"""Frozen BGE CLS encoder for full task instructions, without retrieval prefixes."""

import threading
from pathlib import Path

import torch
import torch.nn.functional as F


class FrozenBgeClsEncoder:
    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.cache = {}
        self.lock = threading.Lock()

    def __deepcopy__(self, memo):
        copied = type(self)(self.model_path)
        memo[id(self)] = copied
        return copied

    @torch.no_grad()
    def encode(self, texts, *, device):
        with self.lock:
            if self.model is None:
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
                self.model = AutoModel.from_pretrained(
                    self.model_path, dtype=torch.float32, local_files_only=True,
                ).eval().requires_grad_(False)
            if self.device != device:
                self.model.to(device)
                self.device = device
                self.cache.clear()
            texts = [str(text) for text in texts]
            with torch.autocast(device_type=device.type, enabled=False):
                for text in dict.fromkeys(texts):
                    if text not in self.cache:
                        tokens = self.tokenizer(text, return_tensors="pt").to(device)
                        cls = self.model(**tokens).last_hidden_state[:, 0].float()
                        self.cache[text] = F.normalize(cls, p=2, dim=-1)[0].cpu()
            return torch.stack([self.cache[text] for text in texts]).to(device)
