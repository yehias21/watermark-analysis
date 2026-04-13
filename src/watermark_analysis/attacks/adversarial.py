"""Embedding-space adversarial PGD attack.

Merges the previously-split ``adv.py`` (encoder models) and ``adversial.py``
(PGD loop). Contains:

- Encoder wrappers around CLIP, ResNet-18 and AutoencoderKL used as the
  similarity target for the attack.
- ``WarmupPGDEmbedding``: PGD in embedding space w/ optional warm-up init.
- ``adv_emb_attack``: convenience entry point that processes a folder of
  watermarked images and writes adversarial outputs.
"""
from __future__ import annotations

import os

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from diffusers.models import AutoencoderKL
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

from ..config import (
    ADV_ALPHA_FACTOR,
    ADV_BATCH_SIZE,
    ADV_DEFAULT_ALPHA,
    ADV_DEFAULT_EPS,
    ADV_EPS_FACTOR,
    ADV_N_STEPS,
    ADV_NUM_WORKERS,
    CLIP_INPUT_SIZE,
    CLIP_MODEL_NAME,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    PNG_EXT,
    RESNET_INPUT_SIZE,
    RESNET_LAYER_MAP,
    RESNET_MEAN,
    RESNET_STD,
)
from .base import Attack


# --------------------------------------------------------------------------- #
# Encoders                                                                    #
# --------------------------------------------------------------------------- #
class BaseEncoder(nn.Module):
    def forward(self, images):
        raise NotImplementedError("This method should be implemented by subclasses.")


class ClipEmbedding(BaseEncoder):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        self.processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.normalizer = transforms.Compose([
            transforms.Resize(CLIP_INPUT_SIZE),
            transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
        ])

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x).cuda())
        return self.model.get_image_features(**inputs)


class VAEEmbedding(BaseEncoder):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_name)

    def forward(self, images):
        images = 2.0 * images - 1.0
        return self.model.encode(images).latent_dist.mode()


class ResNet18Embedding(BaseEncoder):
    def __init__(self, layer: str):
        super().__init__()
        original_model = models.resnet18(pretrained=True)
        if layer not in RESNET_LAYER_MAP:
            raise ValueError("Invalid layer name")
        self.features = nn.Sequential(*list(original_model.children())[:RESNET_LAYER_MAP[layer]])

    def forward(self, images):
        images = TF.resize(images, RESNET_INPUT_SIZE)
        images = (images - RESNET_MEAN) / RESNET_STD
        return self.features(images)


# --------------------------------------------------------------------------- #
# PGD attack                                                                  #
# --------------------------------------------------------------------------- #
class WarmupPGDEmbedding:
    def __init__(
        self,
        model,
        device,
        eps: float = ADV_DEFAULT_EPS,
        alpha: float = ADV_DEFAULT_ALPHA,
        steps: int = 10,
        loss_type: str = "l2",
        random_start: bool = True,
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss_type = loss_type
        self.random_start = random_start
        self.device = device

        if self.loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif self.loss_type == "l2":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")

    def forward(self, images, init_delta=None):
        self.model.eval()
        images = images.clone().detach().to(self.device)
        original_embeddings = self.model(images).detach()

        if self.random_start:
            adv_images = images.clone().detach()
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        elif init_delta is not None:
            clamped_delta = torch.clamp(init_delta, min=-self.eps, max=self.eps)
            adv_images = images.clone().detach() + clamped_delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            assert False

        for _ in tqdm(range(self.steps)):
            self.model.zero_grad()
            adv_images.requires_grad = True
            adv_embeddings = self.model(adv_images)
            cost = self.loss_fn(adv_embeddings, original_embeddings)
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class SimpleImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.filenames = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and os.path.splitext(f)[1].lower() in PNG_EXT
        ]

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.filenames)


def _build_embedding_model(encoder: str):
    if encoder == "resnet18":
        return ResNet18Embedding("last")
    if encoder == "clip":
        return ClipEmbedding()
    if encoder == "klvae16":
        return VAEEmbedding("MudeHui/vae-f16-c16")
    raise ValueError(f"Unsupported encoder: {encoder}")


class AdversarialEmbeddingAttack(Attack):
    """Folder-level adversarial attack implementing the ``Attack`` ABC."""

    def __init__(self, encoder: str, strength: float, device: torch.device | None = None):
        self.encoder = encoder
        self.strength = strength
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def apply(self, image):  # per-image API; full folder impl via run()
        raise NotImplementedError("Use run(input_dir, output_dir) for folder-level operation.")

    def run(self, wm_img_path: str, output_path: str) -> None:
        adv_emb_attack(wm_img_path, self.encoder, self.strength, output_path, device=self.device)


def adv_emb_attack(wm_img_path, encoder, strength, output_path, device=torch.device("cuda")):
    os.makedirs(output_path, exist_ok=True)
    for path in [wm_img_path, output_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path does not exist: {path}")

    embedding_model = _build_embedding_model(encoder).to(device)
    embedding_model.eval()
    print("Embedding Model loaded!")

    transform = transforms.ToTensor()
    wm_dataset = SimpleImageFolder(wm_img_path, transform=transform)
    wm_loader = DataLoader(
        wm_dataset,
        batch_size=ADV_BATCH_SIZE,
        shuffle=False,
        num_workers=ADV_NUM_WORKERS,
        pin_memory=True,
    )
    print("Data loaded!")

    attack = WarmupPGDEmbedding(
        model=embedding_model,
        eps=ADV_EPS_FACTOR * strength,
        alpha=ADV_ALPHA_FACTOR * ADV_EPS_FACTOR * strength,
        steps=ADV_N_STEPS,
        device=device,
    )
    for i, (images, image_paths) in tqdm(enumerate(wm_loader)):
        images = images.to(device)
        print(images.shape)
        print(f"Processing batch {i+1}...")
        images_adv = attack.forward(images)
        for img_adv, image_path in zip(images_adv, image_paths):
            save_path = os.path.join(output_path, os.path.basename(image_path))
            save_image(img_adv, save_path)
    print("Attack finished!")
