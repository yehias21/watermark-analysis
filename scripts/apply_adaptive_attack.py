#!/usr/bin/env python
"""Apply a trained adaptive-VAE attack to a folder of images."""
from __future__ import annotations

import argparse

from watermark_analysis.attacks.adaptive import apply_adaptive_attack


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trained adaptive VAE attack.")
    parser.add_argument("--input-folder", required=True, help="Folder of watermarked PNGs.")
    parser.add_argument("--output-folder", required=True, help="Where to write attacked PNGs.")
    parser.add_argument("--vae-path", required=True, help="Path to the fine-tuned VAE state dict.")
    args = parser.parse_args()

    apply_adaptive_attack(args.input_folder, args.output_folder, args.vae_path)


if __name__ == "__main__":
    main()
