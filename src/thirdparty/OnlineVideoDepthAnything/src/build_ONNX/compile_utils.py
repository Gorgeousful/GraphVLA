from typing import Union, Dict

import torch
import os

from torchvision.transforms import Compose
import numpy as np
from PIL import Image
import cv2
import math
import tensorrt as trt
import pycuda.driver as cuda


def load_dummy_data(h: int, w: int, format: str, cache_size: int = 0, device: str = 'cuda:0') \
        -> Union[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    """
    Prepare cache & dummy data
    --------------------------
    The Cache has dimension: [Number of Temporal Transformer 3DModules, Number of Temporal Attention, batch_size * h * w, Cache_size, Channels]
    Batch_size is in our case always 1 Frame; h and w are depending on the preprocessing resolution which is defined as: 
    Input_h / 14 and Input_w / 14 for Cache1 and Cache3. For the other two caches they are resized to half this resolution (c2) and double this 
    Resolution (c4). 

    Parameters
    ----------
    h : int
        Height of the input image. Used to determine the cache resolutions
    w : int 
        Width of the input image. Used to determine the cache resolutions
    format: Union[str: pytorch, str: numpy]
        The forma the data will be returned in 
    cache_size : int
        Sets the size of the Cache
    device : str
        torch device to put the cache on. 

    Returns
    -------
    inputs : Tuple (inputs, input_cache, mask_indices, input_position)
        Tuple of random inputs to the model and zero initialised cache. 
    """
    assert format in ("pytorch", "numpy")

    c1 = int(h/14 * w/14)
    c2 = int(math.ceil(h/(14*2)) * math.ceil(w/(14*2)))
    c3 = int(h/14 * w/14)
    c4 = int(math.ceil((h * 2)/14) * math.ceil((w * 2)/14))

    input_cache1 = torch.zeros(1, 2, c1, cache_size, 192)
    input_cache2 = torch.zeros(1, 2, c2, cache_size, 384)
    input_cache3 = torch.zeros(1, 2, c3, cache_size, 64)
    input_cache4 = torch.zeros(1, 2, c4, cache_size, 64)

    input_cache = {'c1': input_cache1.to(device) if format == "pytorch" else input_cache1.numpy(),
                   'c2': input_cache2.to(device) if format == "pytorch" else input_cache2.numpy(),
                   'c3': input_cache3.to(device) if format == "pytorch" else input_cache3.numpy(),
                   'c4': input_cache4.to(device) if format == "pytorch" else input_cache4.numpy()  
                   }

    inputs = torch.rand((1, 1, 3, h, w)).to(device) if format == "pytorch" else np.random.rand(1, 1, 3, h, w)
    mask_indices = torch.tensor(list(range(1, cache_size))).to(device) if format == "pytorch" else np.arange(1, cache_size)
    input_position = torch.tensor([cache_size, cache_size]).to(device) if format == "pytorch" else np.array([cache_size, cache_size])

    return (inputs, input_cache, mask_indices, input_position)

def check_cuda_versions():
    """
    Check Cuda versions
    -------------------
    Checks if the cuda versions installed and used for compiling pycuda and tensorrt are the same. 
    If not you have to try to reinstall with matching cuda versions. The scripts will fail if you use 
    different versions. 
    """
    print("--- Checking CUDA versions ---")

    # 1. PyTorch
    if torch.cuda.is_available():
        pytorch_cuda_version = torch.version.cuda
        device_name = torch.cuda.get_device_name(0)
        print(f"PyTorch: ")
        print(f"  - Compiled with cuda version: {pytorch_cuda_version}")
        print(f"  - Current active device: {device_name}")
    else:
        print("PyTorch: CUDA is not available")

    try:
        cuda.init()
        driver_version_int = cuda.get_driver_version()
        major = driver_version_int // 1000
        minor = (driver_version_int % 1000) // 10
        driver_version = f"{major}.{minor}"
        print(f"PyCUDA / NVIDIA Driver:")
        print(f"  - NVIDIA Driver-Version used by pycuda: {driver_version}")
    except Exception as e:
        print(f"PyCUDA: Could not initialise PyCUDA. Error: {e}")

    # 3. TensorRT
    trt_version = trt.__version__
    print(f"TensorRT:")
    print(f"  - TensorRT-Version: {trt_version}")
    
    print("\n--- Check with shell commands ---")
    print("1. NVIDIA driver and CUDA version: `nvidia-smi`")
    print("2. Systhem wide CUDA-Toolkit (if installed): `nvcc --version`")
    print("3. Conda install packages: `conda list | grep cuda`")


if __name__ == "__main__":
    check_cuda_versions()