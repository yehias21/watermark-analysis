"""Shared I/O helpers for listing / loading images.

Deduplicates the ``list_image_files`` / ``ToTensor`` / ``ToPILImage``
patterns that were copy-pasted across the legacy scripts.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .config import DEFAULT_NUM_WORKERS, VALID_IMAGE_EXTENSIONS

TO_TENSOR = transforms.ToTensor()
TO_PIL = transforms.ToPILImage()


def is_image_file(filename: str, extensions: Iterable[str] = VALID_IMAGE_EXTENSIONS) -> bool:
    return os.path.splitext(filename.lower())[1] in set(extensions)


def list_image_files(folder: str, extensions: Iterable[str] = VALID_IMAGE_EXTENSIONS) -> List[str]:
    """Return filenames (not full paths) of images in ``folder``."""
    return [f for f in os.listdir(folder) if is_image_file(f, extensions)]


def list_image_paths(folder: str, extensions: Iterable[str] = VALID_IMAGE_EXTENSIONS) -> List[str]:
    return [os.path.join(folder, f) for f in list_image_files(folder, extensions)]


def load_image(path: str, mode: str | None = None) -> Image.Image:
    img = Image.open(path)
    return img.convert(mode) if mode else img


def load_images_parallel(
    paths: Sequence[str],
    transform=None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    desc: str = "Loading images",
):
    def _load(p):
        img = Image.open(p)
        return transform(img) if transform is not None else img

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        return list(tqdm(ex.map(_load, paths), total=len(paths), desc=desc))


def load_image_tensor_batch(paths: Sequence[str], transform, num_workers: int = DEFAULT_NUM_WORKERS):
    return torch.stack(load_images_parallel(paths, transform=transform, num_workers=num_workers))


def ensure_dir(path: str | Path) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)
