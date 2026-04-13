#!/usr/bin/env python
"""Compare two folders on LPIPS / MSE / PSNR / SSIM / NMI / FID."""
from __future__ import annotations

import argparse

from watermark_analysis.config import DEFAULT_NUM_WORKERS
from watermark_analysis.metrics.quality import compare_folders


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image quality between two folders.")
    parser.add_argument("--ref_folder", required=True)
    parser.add_argument("--target_folder", required=True)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    compare_folders(args.ref_folder, args.target_folder, num_threads=args.num_threads)


if __name__ == "__main__":
    main()
