#!/usr/bin/env python
"""Print N COCO prompts from the shared prompt iterator."""
from __future__ import annotations

import argparse

from watermark_analysis.prompts import next_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit prompts from the COCO iterator.")
    parser.add_argument("--n", type=int, default=10, help="Number of prompts to print.")
    args = parser.parse_args()

    it = next_prompt()
    for _ in range(args.n):
        print(next(it))


if __name__ == "__main__":
    main()
