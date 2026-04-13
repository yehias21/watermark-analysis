"""Watermark-decoding performance metrics.

Combines:

- Script-level logic from the old ``calculate_performance_metrics.py``
  (``WatermarkDecoder`` class, bit-error rate, AUC-ROC, significance
  thresholds).
- Helper functions from the old ``metrics/performance/evasion_rate.py``
  (``bit_error_rate``, ``complex_l1``, ``detection_perforamance`` ROC
  summaries).
- TRW-specific ROC helper from ``trw_perormance_metrics.py``.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from sklearn import metrics as sk_metrics
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm

from ..config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    PERF_ALPHA_01,
    PERF_ALPHA_05,
    PERF_DECISION_THRESHOLD,
    PERFORMANCE_OUTPUT_DIR,
)
from ..watermarks import build_watermark


# --------------------------------------------------------------------------- #
# Low-level helpers (from legacy evasion_rate.py)                             #
# --------------------------------------------------------------------------- #
def bit_error_rate(pred, target):
    if not pred.dtype == target.dtype == bool:
        raise ValueError(f"Cannot compute BER for {pred.dtype} and {target.dtype}")
    return np.mean(pred != target)


def complex_l1(pred, target):
    if not pred.dtype == target.dtype == np.float16:
        raise ValueError(f"Cannot compute Complex L1 for {pred.dtype} and {target.dtype}")
    pred = pred.astype(np.float32).reshape(2, -1)
    target = target.astype(np.float32).reshape(2, -1)
    return np.sqrt(((pred - target) ** 2).sum(0)).mean()


def message_distance(pred, target):
    if target.dtype == bool:
        return bit_error_rate(pred, target)
    if target.dtype == np.float16:
        return complex_l1(pred, target)
    raise TypeError


def detection_performance(original_distances, watermarked_distances):
    if not len(original_distances) == len(watermarked_distances):
        raise ValueError("Length of distances must be equal")
    y_true = [0] * len(original_distances) + [1] * len(watermarked_distances)
    y_score = (-np.array(original_distances + watermarked_distances)).tolist()
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_score, pos_label=1)
    return {
        "acc_1": np.max(1 - (fpr + (1 - tpr)) / 2),
        "auc_1": sk_metrics.auc(fpr, tpr),
        "low100_1": tpr[np.where(fpr < 0.01)[0][-1]],
        "low1000_1": tpr[np.where(fpr < 0.001)[0][-1]],
    }


def mean_and_std(values):
    if values is None:
        return None
    return np.mean(values), np.std(values)


def combine_means_and_stds(ms1, ms2):
    if ms1 is None or ms2 is None:
        return None
    mean1, std1 = ms1
    mean2, std2 = ms2
    return (mean1 + mean2) / 2, np.sqrt((std1**2 + std2**2) / 2)


# --------------------------------------------------------------------------- #
# TRW-specific ROC helper (from legacy trw_perormance_metrics.py)             #
# --------------------------------------------------------------------------- #
def trw_roc_curve(no_w_metrics: Iterable[float], w_metrics: Iterable[float]):
    no_w_metric = list(no_w_metrics)
    w_metric = list(w_metrics)
    true_label = [0] * len(no_w_metric) + [1] * len(w_metric)
    metric = no_w_metric + w_metric
    fpr, tpr, thresholds = roc_curve(true_label, metric)
    return fpr, tpr, thresholds, auc(fpr, tpr)


# --------------------------------------------------------------------------- #
# Script-level decoder (from legacy calculate_performance_metrics.py)         #
# --------------------------------------------------------------------------- #
class WatermarkDecoder:
    def __init__(
        self,
        watermark_algorithm: str = "rivagan",
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ):
        self.watermark_algorithm = watermark_algorithm
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.watermark_key = build_watermark(watermark_algorithm)

    @staticmethod
    def load_single_image(image_path: str) -> Image.Image:
        return Image.open(image_path)

    def load_dataset(self, images_path: str, csv_path: str):
        csv_file = os.path.join(csv_path, "messages.csv")
        data = pd.read_csv(csv_file)
        images_paths = [os.path.join(images_path, f"data_{idx}.png") for idx in data["index"]]

        def load_image_set(paths):
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                return list(tqdm(
                    executor.map(lambda p: self.load_single_image(p), paths),
                    total=len(paths),
                    desc="Loading images",
                ))

        print("Loading images...")
        target_images = load_image_set(images_paths)
        messages = torch.tensor([eval(m) for m in data["message"]])
        return target_images, messages

    def calculate_metrics(self, true_messages, decoded_messages, decoded_probs):
        true_messages = true_messages.cpu()
        decoded_messages = decoded_messages.cpu()
        decoded_probs = decoded_probs.cpu()

        bit_error = float(torch.mean((true_messages != decoded_messages).float()))
        correct_bits_per_message = torch.sum((true_messages == decoded_messages).float(), dim=-1)
        message_length = true_messages.shape[-1]

        critical_value_05 = stats.binom.ppf(1 - PERF_ALPHA_05, message_length, 0.5)
        critical_value_01 = stats.binom.ppf(1 - PERF_ALPHA_01, message_length, 0.5)
        accuracy_05 = float(torch.mean((correct_bits_per_message > critical_value_05).float()))
        accuracy_01 = float(torch.mean((correct_bits_per_message > critical_value_01).float()))

        out = {
            "bit_error_rate": bit_error,
            "accuracy_threshold_0.05": accuracy_05,
            "accuracy_threshold_0.01": accuracy_01,
            "critical_bits_threshold_0.05": float(critical_value_05),
            "critical_bits_threshold_0.01": float(critical_value_01),
            "allowed_error_bits_0.05": float(message_length - critical_value_05),
            "allowed_error_bits_0.01": float(message_length - critical_value_01),
            "auc_roc": float(roc_auc_score(
                true_messages.flatten().numpy(),
                decoded_probs.flatten().numpy(),
            )),
        }
        return out, correct_bits_per_message

    def test_decoding(self, images_path: str, csv_path: str, output_dir: str = PERFORMANCE_OUTPUT_DIR):
        target_images, messages = self.load_dataset(images_path, csv_path)
        num_batches = len(target_images) // self.batch_size
        decoded_messages = []
        decoded_probs = []
        for i in range(num_batches):
            batch_images = target_images[i * self.batch_size:(i + 1) * self.batch_size]
            print(f"Batch {i + 1}/{num_batches}")
            decoded_prob = self.watermark_key.decode(batch_images)
            decoded_messages.append(decoded_prob >= PERF_DECISION_THRESHOLD)
            decoded_probs.append(decoded_prob)

        decoded_messages_t = torch.from_numpy(np.stack(decoded_messages))
        decoded_probs_t = torch.from_numpy(np.stack(decoded_probs))
        messages = messages.view(num_batches, self.batch_size, -1)

        out, correct_bits = self.calculate_metrics(messages, decoded_messages_t, decoded_probs_t)

        print("\nPerformance Metrics:")
        print("-" * 50)
        for name, value in out.items():
            print(f"{name.replace('_', ' ').title()}: {value:.4f}")

        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(out, index=[0]).to_csv(
            os.path.join(output_dir, f"{'_'.join(images_path.split('/')[-2:])}_metrics.csv"),
            index=False,
        )
        return out, correct_bits
