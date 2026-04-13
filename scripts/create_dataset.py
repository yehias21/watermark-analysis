#!/usr/bin/env python
"""Generate either the test dataset or the attack (triplet) training dataset."""
from __future__ import annotations

import argparse

from watermark_analysis.config import DEFAULT_BATCH_SIZE, DEFAULT_NUM_SAMPLES
from watermark_analysis.datasets import create_attack_dataset, create_test_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate watermarking datasets.")
    parser.add_argument("--split", choices=["test", "attack"], required=True,
                        help="'test' for single watermarked images, 'attack' for (no/wm/inv) triplets.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--watermark", type=str, default="dwtdct",
                        choices=["rivagan", "stegastamp", "dwtdct", "dwtdctsvd", "trw"])
    args = parser.parse_args()

    if args.split == "test":
        create_test_dataset(args.num_samples, args.batch_size, args.cache_dir, args.watermark)
    else:
        create_attack_dataset(args.num_samples, args.batch_size, args.cache_dir, args.watermark)


if __name__ == "__main__":
    main()
