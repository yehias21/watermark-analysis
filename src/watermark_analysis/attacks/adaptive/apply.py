"""Apply a trained adaptive VAE attack to a folder of images."""
from __future__ import annotations

import os

import torch
from PIL import Image
from tqdm import tqdm

from ...io import list_image_files
from .model import build_transforms, load_finetuned_vae


@torch.no_grad()
def apply_adaptive_attack(input_folder: str, output_folder: str, vae_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_finetuned_vae(vae_path, device)

    os.makedirs(output_folder, exist_ok=True)
    preprocess, postprocess = build_transforms()

    for filename in tqdm(list_image_files(input_folder), desc="Processing images"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        image = Image.open(input_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        latents = vae.encode(image_tensor).latent_dist.sample()
        reconstructed = vae.decode(latents).sample
        postprocess(reconstructed.squeeze(0).cpu()).save(output_path)
