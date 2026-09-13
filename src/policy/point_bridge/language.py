"""Frozen all-MiniLM-L6-v2 sentence embeddings, using its mean-pool/normalize recipe."""

import torch
from torch.nn import functional as F

from src.policy.language import FrozenBgeClsEncoder


class FrozenMiniLMEncoder(FrozenBgeClsEncoder):
    @torch.no_grad()
    def encode(self, texts, *, device):
        device = torch.device(device)
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
                        tokens = self.tokenizer(text, padding=True, truncation=True,
                                                max_length=256, return_tensors="pt").to(device)
                        hidden = self.model(**tokens).last_hidden_state.float()
                        mask = tokens.attention_mask[..., None].to(hidden.dtype)
                        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
                        self.cache[text] = F.normalize(pooled, dim=-1)[0].cpu()
            return torch.stack([self.cache[text] for text in texts]).to(device)
