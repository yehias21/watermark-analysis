"""Unified dataset-generation entry points (merges legacy create_test_dataset
and create_train_data_test modules)."""
from __future__ import annotations

import os

import pandas as pd
import torch
import tqdm

from .config import (
    ATTACK_DATASET_SEED,
    BIT_ALGORITHMS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_SAMPLES,
    RIVAGAN_DWT_MSG_LEN,
    SD_MODEL_V1_5,
    STEGASTAMP_MSG_LEN,
    TEST_DATASET_SEED,
    TRW_MSG_LEN,
)
from .prompts import next_prompt
from .watermarks.post_processing_sd import PostProccessingWatermarksStableDiffusion
from .watermarks.trw_stable_diffusion import TrwStableDiffusion


def _build_generator(watermark_algorthim: str, model: str | None = None):
    if watermark_algorthim == "trw":
        return TrwStableDiffusion()
    if model is not None:
        return PostProccessingWatermarksStableDiffusion(model=model, watermark_algorthim=watermark_algorthim)
    return PostProccessingWatermarksStableDiffusion(watermark_algorthim=watermark_algorthim)


def _sample_messages_test(generator, watermark_algorthim: str, num_prompts: int):
    if watermark_algorthim == "trw":
        generator.trw.set_message(generator.trw.sample_message(TRW_MSG_LEN).squeeze(0))
        return generator.trw.get_message().to(generator.device)
    if watermark_algorthim in BIT_ALGORITHMS:
        return torch.randint(0, 2, (num_prompts, RIVAGAN_DWT_MSG_LEN)).float()
    if watermark_algorthim == "stegastamp":
        return torch.randint(0, 2, (num_prompts, STEGASTAMP_MSG_LEN)).float()
    return None


def _sample_messages_attack(generator, watermark_algorthim: str, num_prompts: int):
    if watermark_algorthim == "trw":
        generator.trw.set_message(generator.trw.sample_message(1)[0])
        return torch.repeat_interleave(
            generator.trw.get_message().to(generator.device).unsqueeze(0),
            num_prompts,
            dim=0,
        )
    if watermark_algorthim in BIT_ALGORITHMS:
        return torch.randint(0, 2, (num_prompts, RIVAGAN_DWT_MSG_LEN)).float()
    if watermark_algorthim == "stegastamp":
        return torch.randint(0, 2, (num_prompts, STEGASTAMP_MSG_LEN)).float()
    return None


def _seed(s: int) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def create_test_dataset(
    number_of_samples: int = DEFAULT_NUM_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_dir: str = "cache",
    watermark_algorthim: str = "dwtdctsvd",
    seed: int = TEST_DATASET_SEED,
) -> None:
    _seed(seed)
    prompt_iterator = next_prompt()
    cache_dir_watermark = os.path.join(cache_dir, f"test_dataset_{watermark_algorthim}")
    os.makedirs(cache_dir_watermark, exist_ok=True)

    generator = _build_generator(watermark_algorthim)
    tqdm_bar = tqdm.tqdm(range(number_of_samples // batch_size), desc="Generating Test Dataset")
    data_acc = []
    image_index = 0

    for _ in tqdm_bar:
        prompts = [next(prompt_iterator) for _ in range(batch_size)]
        messages = _sample_messages_test(generator, watermark_algorthim, len(prompts))
        images = generator.generate(prompts, watermark=True, messages=messages)
        for image, prompt, message in zip(images, prompts, messages):
            image.save(os.path.join(cache_dir_watermark, f"data_{image_index}.png"))
            data_acc.append({"index": image_index, "prompt": prompt, "message": message.tolist()})
            image_index += 1

    pd.DataFrame(data_acc).to_csv(os.path.join(cache_dir_watermark, "messages.csv"))
    print("Test Dataset Created")


@torch.no_grad()
def create_attack_dataset(
    number_of_samples: int = DEFAULT_NUM_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_dir: str = "cache",
    watermark_algorthim: str = "dwtdct",
    seed: int = ATTACK_DATASET_SEED,
) -> None:
    _seed(seed)
    prompt_iterator = next_prompt()
    base_dir = os.path.join(cache_dir, f"attack_dataset_{watermark_algorthim}")
    for sub in ("no_watermark", "watermark", "inverse_watermark"):
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    generator = _build_generator(watermark_algorthim, model=SD_MODEL_V1_5)
    tqdm_bar = tqdm.tqdm(range(number_of_samples // batch_size), desc="Generating Attack Dataset")
    data_acc = []
    image_index = 0

    for _ in tqdm_bar:
        prompts = [next(prompt_iterator) for _ in range(batch_size)]
        messages = _sample_messages_attack(generator, watermark_algorthim, len(prompts))
        images_no, images_wm, images_inv = generator.generate_triplet(prompts, messages)
        for no_wm, wm, inv, prompt, message in zip(images_no, images_wm, images_inv, prompts, messages):
            no_wm.save(os.path.join(base_dir, "no_watermark", f"data_{image_index}.png"))
            wm.save(os.path.join(base_dir, "watermark", f"data_{image_index}.png"))
            inv.save(os.path.join(base_dir, "inverse_watermark", f"data_{image_index}.png"))
            data_acc.append({"index": image_index, "prompt": prompt, "message": message.tolist()})
            image_index += 1

    pd.DataFrame(data_acc).to_csv(os.path.join(base_dir, "messages.csv"))
    print("Attack Dataset Created")
