"""COCO-caption prompt iterator used as Stable Diffusion input."""
from __future__ import annotations

import json
import random

from .config import COCO_PATH, PROMPTS_SEED


def next_prompt(coco_path: str | None = None, seed: int = PROMPTS_SEED):
    """Generator yielding shuffled COCO captions as prompts."""
    random.seed(seed)
    with open(coco_path or COCO_PATH) as f:
        dataset = json.load(f)["annotations"]
    random.shuffle(dataset)
    for row in dataset:
        yield row["caption"]
