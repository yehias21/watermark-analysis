#!/usr/bin/env python
"""Inspect ONNX model inputs/outputs."""
from __future__ import annotations

import argparse
import os

import onnxruntime as ort

from watermark_analysis.config import (
    RIVAGAN_DECODER_PATH,
    RIVAGAN_ENCODER_PATH,
    STEGASTAMP_MODEL_PATH,
)


def inspect_onnx_operations(model_path: str) -> None:
    session = ort.InferenceSession(model_path)
    print(f"\n== {model_path} ==")
    print("Inputs:")
    for i in session.get_inputs():
        print(f"- Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
    print("Outputs:")
    for o in session.get_outputs():
        print(f"- Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
    print("Providers:", session.get_providers())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect bundled watermark ONNX models.")
    parser.add_argument("--model", nargs="*", default=None,
                        help="ONNX file(s) to inspect. Defaults to the three bundled models.")
    args = parser.parse_args()

    paths = args.model or [RIVAGAN_ENCODER_PATH, RIVAGAN_DECODER_PATH, STEGASTAMP_MODEL_PATH]
    for p in paths:
        if not os.path.exists(p):
            print(f"[skip] not found: {p}")
            continue
        inspect_onnx_operations(p)


if __name__ == "__main__":
    main()
