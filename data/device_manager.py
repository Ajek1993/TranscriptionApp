#!/usr/bin/env python3
"""
Device detection and management for GPU/CPU selection.

This module provides functionality for:
- Detecting available devices (CUDA GPU vs CPU)
- Validating cuDNN compatibility
- Monitoring GPU memory usage
"""

from typing import Tuple
from tqdm import tqdm


# Memory requirements for Whisper models (in GB)
WHISPER_MODEL_MEMORY_REQUIREMENTS = {
    'tiny': 1, 'base': 1, 'small': 2, 'medium': 5, 'large': 10,
    'large-v2': 10, 'large-v3': 10
}


def get_gpu_memory_info() -> str:
    """
    Get GPU memory usage information.

    Returns:
        String with GPU memory stats, or empty string if GPU not available
    """
    try:
        import torch
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            free = total_mem - allocated
            return f"GPU Memory: {allocated:.2f}GB/{total_mem:.2f}GB ({free:.2f}GB free)"
    except:
        pass
    return ""


def check_vram_for_model(model_name: str) -> Tuple[bool, str]:
    """
    Sprawdza czy dostępna VRAM wystarczy dla danego modelu.

    Args:
        model_name: Nazwa modelu Whisper (tiny, base, small, medium, large, etc.)

    Returns:
        Tuple (czy_wystarczy: bool, komunikat_ostrzeżenia: str)
        - czy_wystarczy: True jeśli VRAM wystarcza lub GPU niedostępne
        - komunikat_ostrzeżenia: Pusty string jeśli OK, ostrzeżenie jeśli nie
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return True, ""  # Bez GPU nie sprawdzamy VRAM

        # Pobierz wymagania modelu
        required_gb = WHISPER_MODEL_MEMORY_REQUIREMENTS.get(model_name, 0)
        if required_gb == 0:
            return True, ""  # Nieznany model - nie sprawdzamy

        # Pobierz dostępną pamięć GPU
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        free_gb = total_mem - allocated

        if free_gb < required_gb:
            # Sugeruj mniejszy model
            smaller_models = []
            for name, req in WHISPER_MODEL_MEMORY_REQUIREMENTS.items():
                if req < required_gb and req <= free_gb:
                    smaller_models.append((name, req))

            suggestion = ""
            if smaller_models:
                # Wybierz największy model który się zmieści
                best = max(smaller_models, key=lambda x: x[1])
                suggestion = f"\n          Rozważ użycie mniejszego modelu (--model {best[0]} wymaga ~{best[1]}GB)."

            warning = (
                f"Model '{model_name}' wymaga ~{required_gb}GB VRAM, dostępne: {free_gb:.1f}GB.\n"
                f"          Może wystąpić błąd 'out of memory'.{suggestion}"
            )
            return False, warning

        return True, ""

    except Exception:
        return True, ""  # W razie błędu - kontynuuj bez ostrzeżenia


def detect_device(force_device: str = 'auto') -> Tuple[str, str]:
    """
    Detect available device with cuDNN validation.

    Args:
        force_device: Device override ('auto', 'cuda', 'cpu')

    Returns:
        Tuple of (device: str, device_info: str)
    """
    # Force CPU if requested
    if force_device == 'cpu':
        return "cpu", "CPU (forced by --device cpu)"

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()

            if cudnn_version and cudnn_version >= 8000:  # cuDNN 8.x
                if force_device == 'cuda':
                    device_info = f"NVIDIA GPU ({gpu_name}, CUDA {cuda_version}, cuDNN {cudnn_version // 1000}.{(cudnn_version % 1000) // 100}) [forced]"
                else:
                    device_info = f"NVIDIA GPU ({gpu_name}, CUDA {cuda_version}, cuDNN {cudnn_version // 1000}.{(cudnn_version % 1000) // 100})"
                return "cuda", device_info
            else:
                if force_device == 'cuda':
                    tqdm.write(f"Warning: --device cuda requested but cuDNN {cudnn_version} too old. Need >=8.0. Falling back to CPU")
                else:
                    tqdm.write(f"Warning: cuDNN {cudnn_version} too old. Need >=8.0. Falling back to CPU")
                return "cpu", f"CPU (cuDNN incompatible: {cudnn_version})"
        else:
            # No CUDA available
            if force_device == 'cuda':
                tqdm.write("Warning: --device cuda requested but CUDA not available. Falling back to CPU")
            return "cpu", "CPU (CUDA not available)"
    except ImportError:
        if force_device == 'cuda':
            tqdm.write("Warning: --device cuda requested but PyTorch not available. Falling back to CPU")
        pass
    except Exception as e:
        tqdm.write(f"Warning: GPU detection failed: {e}")
        if force_device == 'cuda':
            tqdm.write("Falling back to CPU")

    return "cpu", "CPU"
