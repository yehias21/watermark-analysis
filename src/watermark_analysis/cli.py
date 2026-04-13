"""Console-script entry points (referenced from ``pyproject.toml``).

Each function parses its own args and dispatches to the relevant package
module. The thin wrappers under ``scripts/`` just forward to these.
"""
from __future__ import annotations

import argparse
import os

from .config import DEFAULT_BATCH_SIZE, DEFAULT_NUM_SAMPLES, DEFAULT_NUM_WORKERS


def create_dataset_main() -> None:
    from .datasets import create_attack_dataset, create_test_dataset

    p = argparse.ArgumentParser(description="Generate watermarking datasets.")
    p.add_argument("--split", choices=["test", "attack"], required=True)
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--cache-dir", type=str, default="cache")
    p.add_argument("--watermark", type=str, default="dwtdct",
                   choices=["rivagan", "stegastamp", "dwtdct", "dwtdctsvd", "trw"])
    a = p.parse_args()
    (create_test_dataset if a.split == "test" else create_attack_dataset)(
        a.num_samples, a.batch_size, a.cache_dir, a.watermark,
    )


def train_adaptive_main() -> None:
    from .attacks.adaptive import train_adaptive_attack

    p = argparse.ArgumentParser(description="Train the adaptive VAE attack.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--cache-dir", type=str, default="cache/attack_dataset_dwtdct")
    p.add_argument("--mode", choices=["inverse_watermark", "no_watermark"], default="no_watermark")
    a = p.parse_args()
    train_adaptive_attack(
        batch_size=a.batch_size, num_epochs=a.num_epochs, learning_rate=a.learning_rate,
        alpha=a.alpha, beta=a.beta, cache_dir=a.cache_dir, mode=a.mode,
    )


def apply_adaptive_main() -> None:
    from .attacks.adaptive import apply_adaptive_attack

    p = argparse.ArgumentParser(description="Run the trained adaptive VAE attack.")
    p.add_argument("--input-folder", required=True)
    p.add_argument("--output-folder", required=True)
    p.add_argument("--vae-path", required=True)
    a = p.parse_args()
    apply_adaptive_attack(a.input_folder, a.output_folder, a.vae_path)


def eval_performance_main() -> None:
    from .metrics.performance import WatermarkDecoder

    p = argparse.ArgumentParser(description="Evaluate watermark-decoding performance.")
    p.add_argument("--images_path", required=True)
    p.add_argument("--csv_path", required=True)
    p.add_argument("--algorithm", required=True,
                   choices=["rivagan", "stegastamp", "dwtdct", "dwtdctsvd"])
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    a = p.parse_args()
    WatermarkDecoder(
        watermark_algorithm=a.algorithm, batch_size=a.batch_size, num_workers=a.num_workers,
    ).test_decoding(a.images_path, a.csv_path)


def eval_quality_main() -> None:
    from .metrics.quality import compare_folders

    p = argparse.ArgumentParser(description="Evaluate image quality between two folders.")
    p.add_argument("--ref_folder", required=True)
    p.add_argument("--target_folder", required=True)
    p.add_argument("--num-threads", type=int, default=DEFAULT_NUM_WORKERS)
    a = p.parse_args()
    compare_folders(a.ref_folder, a.target_folder, num_threads=a.num_threads)


def inspect_onnx_main() -> None:
    import onnxruntime as ort

    from .config import RIVAGAN_DECODER_PATH, RIVAGAN_ENCODER_PATH, STEGASTAMP_MODEL_PATH

    p = argparse.ArgumentParser(description="Inspect bundled watermark ONNX models.")
    p.add_argument("--model", nargs="*", default=None)
    a = p.parse_args()
    paths = a.model or [RIVAGAN_ENCODER_PATH, RIVAGAN_DECODER_PATH, STEGASTAMP_MODEL_PATH]
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] not found: {path}")
            continue
        session = ort.InferenceSession(path)
        print(f"\n== {path} ==")
        for i in session.get_inputs():
            print(f"in: {i.name} {i.shape} {i.type}")
        for o in session.get_outputs():
            print(f"out: {o.name} {o.shape} {o.type}")
        print("providers:", session.get_providers())


def generate_prompts_main() -> None:
    from .prompts import next_prompt

    p = argparse.ArgumentParser(description="Emit prompts from the COCO iterator.")
    p.add_argument("--n", type=int, default=10)
    a = p.parse_args()
    it = next_prompt()
    for _ in range(a.n):
        print(next(it))
