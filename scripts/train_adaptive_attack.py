#!/usr/bin/env python
"""Fine-tune the SDXL-Refiner VAE against a specific watermark."""
from __future__ import annotations

import argparse

from watermark_analysis.attacks.adaptive import train_adaptive_attack


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the adaptive VAE attack.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--cache-dir", type=str, default="cache/attack_dataset_dwtdct")
    parser.add_argument("--mode", choices=["inverse_watermark", "no_watermark"], default="no_watermark")
    args = parser.parse_args()

    train_adaptive_attack(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        beta=args.beta,
        cache_dir=args.cache_dir,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
