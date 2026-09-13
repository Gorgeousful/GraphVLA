"""Flat observation messages: a small JSON header followed by raw NumPy arrays."""

import json
import math
import struct

import numpy as np


def encode_observation(request: dict) -> bytes:
    metadata, arrays, buffers = {}, {}, []
    for key, value in request.items():
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "buif":
                raise ValueError(f"Unsupported observation dtype: {value.dtype}")
            arrays[key] = {"dtype": value.dtype.str, "shape": list(value.shape)}
            value = np.ascontiguousarray(value)
            buffers.append(value.tobytes())
        else:
            metadata[key] = value
    header = json.dumps({"metadata": metadata, "arrays": arrays}).encode("utf-8")
    return b"GVO1" + struct.pack("!I", len(header)) + header + b"".join(buffers)


def decode_observation(message: bytes | str) -> dict:
    # Text requests remain readable by diagnostic tools and existing clients.
    if isinstance(message, str):
        return json.loads(message)
    if len(message) < 8 or message[:4] != b"GVO1":
        raise ValueError("Invalid binary observation header")
    header_size = struct.unpack("!I", message[4:8])[0]
    offset = 8 + header_size
    if offset > len(message):
        raise ValueError("Truncated observation header")
    header = json.loads(message[8:offset])
    result = header["metadata"]
    for key, spec in header["arrays"].items():
        dtype, shape = np.dtype(spec["dtype"]), spec["shape"]
        if dtype.kind not in "buif" or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Invalid observation array dtype or shape")
        size = math.prod(shape) * dtype.itemsize
        if key in result or offset + size > len(message):
            raise ValueError("Duplicate field or truncated observation array")
        result[key] = np.frombuffer(message, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape).copy()
        offset += size
    if offset != len(message):
        raise ValueError("Unexpected trailing observation bytes")
    return result
