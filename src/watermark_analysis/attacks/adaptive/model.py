"""VAE construction + pre/post-processing transforms for the adaptive attack.

Factored out from the legacy ``train_new_adaptive_attack.py`` /
``apply_new_adaptive_attack.py`` scripts.
"""
from __future__ import annotations

from typing import Tuple

import torch
from diffusers import AutoencoderKL
from torchvision import transforms

from ...config import ADAPTIVE_IMAGE_SIZE, SDXL_REFINER_MODEL


def load_pretrained_vae(device: torch.device) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(
        SDXL_REFINER_MODEL,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.enable_gradient_checkpointing()
    return vae


def load_finetuned_vae(vae_path: str, device: torch.device) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(SDXL_REFINER_MODEL, subfolder="vae", torch_dtype=torch.float32)
    vae.load_state_dict(torch.load(vae_path))
    vae.to(device)
    vae.eval()
    return vae


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    preprocess = transforms.Compose([
        transforms.Resize(ADAPTIVE_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # to [-1, 1]
    ])
    postprocess = transforms.Compose([
        transforms.Normalize([-1], [2]),     # back to [0, 1]
        transforms.Lambda(lambda x: x.clamp(0, 1)),
        transforms.ToPILImage(),
    ])
    return preprocess, postprocess


def build_loading_transform() -> transforms.Compose:
    """Transform used when bulk-loading the attack training dataset."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(ADAPTIVE_IMAGE_SIZE, antialias=True),
    ])
