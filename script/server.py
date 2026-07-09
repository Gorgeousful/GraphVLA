#!/usr/bin/env python3
"""Minimal GraphVLA inference server.

The server runs input -> model.infer -> configured out_transforms.
Benchmark-specific executable action conversion belongs to the client side.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from rich.console import Console

from src.model.model import PointQueryModel
from src.training.checkpoint import TrainingCheckpoint

cs = Console()


class InferenceModel:
    """Model wrapper with optional BGE text condition encoding."""

    def __init__(
        self,
        *,
        model_kwargs: Mapping[str, Any],
        data_kwargs: Mapping[str, Any] | None = None,
        ckpt_path: str | Path,
        bge_path: str | Path | None = None,
        device: torch.device,
    ) -> None:
        self.device = device
        self.out_transforms = tuple((data_kwargs or {}).get("out_transforms", ()))
        self.bge_path = Path(bge_path) if bge_path is not None else Path(
            "/data0/luokang/dataset/luokang/ckpts/bge-small-en-v1.5"
        )
        self.bge_tokenizer = None
        self.bge_model = None
        self.embedding_cache: dict[str, torch.Tensor] = {}

        self.model = PointQueryModel(**dict(model_kwargs)).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        incompatible = self.model.load_state_dict(TrainingCheckpoint.unwrap_model_state(state), strict=False)
        cs.print(
            f"[green]loaded checkpoint from {ckpt_path}[/green] "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
        self.model.eval()

    @torch.inference_mode()
    def infer(self, input_data: Mapping[str, Any]) -> dict[str, Any]:
        def tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(input_data[name], device=self.device).to(dtype=dtype)

        def optional_tensor(name: str, dtype: torch.dtype) -> torch.Tensor | None:
            if name not in input_data or input_data[name] is None:
                return None
            return torch.as_tensor(input_data[name], device=self.device).to(dtype=dtype)

        object_condition = optional_tensor("object_condition", torch.float32)
        if object_condition is None and input_data.get("object_condition_texts") is not None:
            text_conditions = input_data["object_condition_texts"]
            if not isinstance(text_conditions, Sequence) or isinstance(text_conditions, str | bytes):
                raise TypeError("object_condition_texts must be a list")
            object_condition = torch.stack([self.encode_condition(item) for item in text_conditions], dim=0)

        actor_condition = optional_tensor("actor_condition", torch.float32)
        if actor_condition is None and input_data.get("actor_condition_text") is not None:
            actor_condition = self.encode_condition(input_data["actor_condition_text"]).unsqueeze(0)

        infer_inputs = {
            "point_feats": tensor("point_feats", torch.float32),
            "actor_feats": tensor("actor_feats", torch.float32),
            "object_id": tensor("object_id", torch.long),
            "point_id": tensor("point_id", torch.long),
            "frame_id": tensor("frame_id", torch.long),
            "object_condition": object_condition,
            "actor_condition": actor_condition,
            "query_type": optional_tensor("query_type", torch.long),
            "head_names": input_data.get("head_names"),
            "frame_query_frame_id": optional_tensor("frame_query_frame_id", torch.long),
            "frame_query_type": optional_tensor("frame_query_type", torch.long),
            "frame_head_names": input_data.get("frame_head_names"),
        }
        outputs = self.model.infer(**infer_inputs)
        output_data = {"outputs": outputs, "batch": infer_inputs}
        for transform in self.out_transforms:
            output_data = transform(output_data)
        return self.to_json(output_data["outputs"])

    def encode_condition(self, value: Any) -> torch.Tensor:
        if isinstance(value, Mapping):
            parts = [value.get("role"), value.get("action_type"), value.get("action_degree")]
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            if len(value) > 3:
                raise ValueError("text condition sequence must have at most 3 items")
            parts = list(value) + [None] * (3 - len(value))
        else:
            raise TypeError("text condition must be a dict or [role, action_type, action_degree]")
        return torch.cat([self.embed_text(None if part is None else str(part)) for part in parts], dim=0)

    def embed_text(self, text: str | None) -> torch.Tensor:
        if self.bge_model is None:
            from transformers import AutoModel, AutoTokenizer

            self.bge_tokenizer = AutoTokenizer.from_pretrained(self.bge_path)
            self.bge_model = AutoModel.from_pretrained(self.bge_path).to(self.device)
            self.bge_model.eval()
            cs.print(f"[green]loaded BGE from {self.bge_path}[/green]")

        assert self.bge_tokenizer is not None and self.bge_model is not None
        dim = int(self.bge_model.config.hidden_size)
        if text is None or text == "":
            return torch.zeros(dim, dtype=torch.float32, device=self.device)
        if text not in self.embedding_cache:
            batch = self.bge_tokenizer([text], padding=True, truncation=True, return_tensors="pt")
            batch = {key: value.to(self.device) for key, value in batch.items()}
            output = self.bge_model(**batch)
            embedding = F.normalize(output.last_hidden_state[:, 0], p=2, dim=1)[0]
            self.embedding_cache[text] = embedding.detach().float()
        return self.embedding_cache[text].to(self.device)

    def to_json(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Mapping):
            return {key: self.to_json(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [self.to_json(item) for item in value]
        if isinstance(value, list):
            return [self.to_json(item) for item in value]
        return value


class InferenceServer:
    """HTTP layer for InferenceModel."""

    def __init__(self, inference: InferenceModel, *, host: str, port: int) -> None:
        self.inference = inference
        self.host = host
        self.port = port
        self.httpd = ThreadingHTTPServer((host, port), self.handler_class())

    def serve_forever(self) -> None:
        cs.print(f"[green]GraphVLA inference server listening on http://{self.host}:{self.port}[/green]")
        cs.print("POST /infer expects normalized model inputs and returns transformed model outputs.")
        self.httpd.serve_forever()

    def handler_class(self) -> type[BaseHTTPRequestHandler]:
        inference = self.inference

        class Handler(BaseHTTPRequestHandler):
            server_version = "GraphVLAInfer/0.1"

            def do_GET(self) -> None:
                self.write_json({"status": "ok"} if self.path == "/health" else {"error": "not found"}, status=200 if self.path == "/health" else 404)

            def do_POST(self) -> None:
                if self.path != "/infer":
                    self.write_json({"error": "not found"}, status=404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    input_data = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(input_data, Mapping):
                        raise TypeError("request body must be a JSON object")
                    self.write_json({"outputs": inference.infer(input_data)})
                except Exception as exc:
                    self.write_json({"error": str(exc)}, status=400)

            def write_json(self, input_data: Mapping[str, Any], *, status: int = 200) -> None:
                body = json.dumps(input_data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: Any) -> None:
                cs.print(f"{self.address_string()} - {fmt % args}")

        return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve GraphVLA model.infer over HTTP.")
    parser.add_argument("--example", default="libero", choices=("libero",))
    parser.add_argument("--ckpt-path", required=True, help="Path to a training checkpoint.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10092)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--bge-path",
        default="/data0/luokang/dataset/luokang/ckpts/bge-small-en-v1.5",
        help="Path to the BGE model used for text conditions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.example == "libero":
        from examples.libero.config.data_config import LIBERO_DATA_CONFIG
        from examples.libero.config.model_config import LIBERO_MODEL_CONFIG

        model_kwargs = LIBERO_MODEL_CONFIG.to_kwargs()
        data_kwargs = LIBERO_DATA_CONFIG.to_kwargs()
    else:
        raise ValueError(f"Unsupported example: {args.example}")

    inference = InferenceModel(
        model_kwargs=model_kwargs,
        data_kwargs=data_kwargs,
        ckpt_path=args.ckpt_path,
        device=torch.device(args.device),
        bge_path=args.bge_path,
    )
    server = InferenceServer(inference, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
