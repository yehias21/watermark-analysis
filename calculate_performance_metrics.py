import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from watermarks.DwtDct import DwtDCT
from watermarks.Rivagan import Rivagan
from watermarks.StegaStamp import StegaStamp

# Module-level constants
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 24
DECISION_THRESHOLD = 0.5
ALPHA_05 = 0.05
ALPHA_01 = 0.01
OUTPUT_DIR = './performance'


class WatermarkDecoder:
    def __init__(self, watermark_algorithm: str = 'rivagan',
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 num_workers: int = DEFAULT_NUM_WORKERS):
        self.watermark_algorithm = watermark_algorithm
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.watermark_key = self._initialize_watermark()

    def _initialize_watermark(self):
        if self.watermark_algorithm == 'rivagan':
            return Rivagan()
        elif self.watermark_algorithm == 'stegastamp':
            return StegaStamp()
        elif self.watermark_algorithm == 'dwtdct':
            return DwtDCT(use_svd=False)
        elif self.watermark_algorithm == 'dwtdctsvd':
            return DwtDCT(use_svd=True)

    @staticmethod
    def load_single_image(image_path: str) -> Image.Image:
        return Image.open(image_path)

    def load_dataset(self, images_path: str, csv_path: str):
        csv_file = os.path.join(csv_path, 'messages.csv')
        data = pd.read_csv(csv_file)

        images_paths = [os.path.join(images_path, f'data_{idx}.png') for idx in data['index']]

        def load_image_set(paths):
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                images = list(tqdm(
                    executor.map(lambda p: self.load_single_image(p), paths),
                    total=len(paths),
                    desc="Loading images"
                ))
            return images

        print("Loading images...")
        target_images = load_image_set(images_paths)
        messages = torch.tensor([eval(m) for m in data['message']])

        return target_images, messages

    def calculate_metrics(self, true_messages, decoded_messages, decoded_probs):
        true_messages = true_messages.cpu()
        decoded_messages = decoded_messages.cpu()
        decoded_probs = decoded_probs.cpu()

        # Calculate overall bit error rate
        bit_error_rate = float(torch.mean((true_messages != decoded_messages).float()))

        # Calculate number of correct bits per message
        correct_bits_per_message = torch.sum((true_messages == decoded_messages).float(), dim=-1)
        message_length = true_messages.shape[-1]

        # Calculate critical values for both alpha levels
        critical_value_05 = stats.binom.ppf(1 - ALPHA_05, message_length, 0.5)
        critical_value_01 = stats.binom.ppf(1 - ALPHA_01, message_length, 0.5)

        # Calculate accuracy for both thresholds
        significant_messages_05 = (correct_bits_per_message > critical_value_05).float()
        significant_messages_01 = (correct_bits_per_message > critical_value_01).float()

        accuracy_threshold_05 = float(torch.mean(significant_messages_05))
        accuracy_threshold_01 = float(torch.mean(significant_messages_01))

        metrics = {
            'bit_error_rate': bit_error_rate,
            'accuracy_threshold_0.05': accuracy_threshold_05,
            'accuracy_threshold_0.01': accuracy_threshold_01,
            'critical_bits_threshold_0.05': float(critical_value_05),
            'critical_bits_threshold_0.01': float(critical_value_01),
            'allowed_error_bits_0.05': float(message_length - critical_value_05),
            'allowed_error_bits_0.01': float(message_length - critical_value_01),
        }

        metrics['auc_roc'] = float(roc_auc_score(
            true_messages.flatten().numpy(),
            decoded_probs.flatten().numpy()
        ))

        return metrics, correct_bits_per_message

    def test_decoding(self, images_path: str, csv_path: str):
        target_images, messages = self.load_dataset(images_path, csv_path)
        num_batches = len(target_images) // self.batch_size
        decoded_messages = []
        decoded_probs = []
        for i in range(num_batches):
            batch_images = target_images[i * self.batch_size:(i + 1) * self.batch_size]
            print(f"Batch {i + 1}/{num_batches}")
            decoded_prob = self.watermark_key.decode(batch_images)
            decoded_messages.append(decoded_prob >= DECISION_THRESHOLD)
            decoded_probs.append(decoded_prob)

        decoded_messages = torch.from_numpy(np.stack(decoded_messages))
        decoded_probs = torch.from_numpy(np.stack(decoded_probs))
        messages = messages.view(num_batches, self.batch_size, -1)

        metrics, correct_bits = self.calculate_metrics(messages, decoded_messages, decoded_probs)

        # Print metrics
        print("\nPerformance Metrics:")
        print("-" * 50)
        for metric_name, value in metrics.items():
            print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")

        # save csv
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(
            os.path.join(OUTPUT_DIR, f"{'_'.join(images_path.split('/')[-2:])}_metrics.csv"),
            index=False,
        )

        return metrics, correct_bits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test watermark decoding on images')
    parser.add_argument('--images_path', type=str, required=True, help='Path to the images folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV folder')
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['rivagan', 'stegastamp', 'dwtdct', 'dwtdctsvd'],
                        help='Watermark algorithm to use')
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    decoder = WatermarkDecoder(
        watermark_algorithm=args.algorithm,
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=DEFAULT_NUM_WORKERS,
    )

    metrics, correct_bits = decoder.test_decoding(args.images_path, args.csv_path)
