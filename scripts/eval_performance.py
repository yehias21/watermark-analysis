#!/usr/bin/env python
"""Compute watermark-decoding performance metrics for a folder of images."""
from __future__ import annotations

import argparse

from watermark_analysis.config import DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS
from watermark_analysis.metrics.performance import WatermarkDecoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate watermark-decoding performance.")
    parser.add_argument("--images_path", required=True)
    parser.add_argument("--csv_path", required=True, help="Folder containing messages.csv.")
    parser.add_argument("--algorithm", required=True,
                        choices=["rivagan", "stegastamp", "dwtdct", "dwtdctsvd"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    decoder = WatermarkDecoder(
        watermark_algorithm=args.algorithm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    decoder.test_decoding(args.images_path, args.csv_path)


if __name__ == "__main__":
    main()
