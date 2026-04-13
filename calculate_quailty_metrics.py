import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_fid import fid_score
from skimage.metrics import (
    normalized_mutual_information,
    peak_signal_noise_ratio,
    structural_similarity,
)
from torchvision import transforms

# Module-level constants
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
FID_BATCH_SIZE = 50
FID_DIMS = 2048
DEFAULT_NUM_THREADS = 24
OUTPUT_DIR = './quality'
METRIC_KEYS = ('LPIPS', 'MSE', 'PSNR', 'SSIM', 'NMI')


class ImageComparator:
    ALLOWED_IMAGE_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS

    def __init__(self, folder1: str, folder2: str):
        self.folder1 = folder1
        self.folder2 = folder2
        self.metrics = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def is_image_file(filename: str) -> bool:
        """Check if a file is an image based on its extension."""
        return os.path.splitext(filename.lower())[1] in ALLOWED_IMAGE_EXTENSIONS

    def calculate_metrics(self, img1_path: str, img2_path: str) -> Optional[dict]:
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            img1_np = np.array(img1)
            img2_np = np.array(img2)

            img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
            img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)

            return {
                'LPIPS': float(self.lpips_fn(img1_tensor, img2_tensor).item()),
                'MSE': np.mean((img1_np - img2_np) ** 2),
                'PSNR': peak_signal_noise_ratio(img1_np, img2_np),
                'SSIM': structural_similarity(img1_np, img2_np, channel_axis=2),
                'NMI': normalized_mutual_information(img1_np, img2_np),
            }
        except Exception as e:
            print(f"Error processing images {img1_path} and {img2_path}: {str(e)}")
            return None

    def process_image_pair(self, img_name: str) -> Optional[dict]:
        img1_path = os.path.join(self.folder1, img_name)
        img2_path = os.path.join(self.folder2, img_name)

        if os.path.exists(img1_path) and os.path.exists(img2_path):
            return self.calculate_metrics(img1_path, img2_path)
        return None

    def calculate_fid(self) -> float:
        return fid_score.calculate_fid_given_paths(
            [self.folder1, self.folder2],
            FID_BATCH_SIZE,
            self.device,
            FID_DIMS,
        )

    def get_image_files(self, folder: str) -> List[str]:
        """Get list of image files in a folder."""
        return [f for f in os.listdir(folder) if self.is_image_file(f)]

    def run_comparison(self, num_threads: int = 4) -> dict:
        images1 = set(self.get_image_files(self.folder1))
        images2 = set(self.get_image_files(self.folder2))
        common_images = list(images1.intersection(images2))

        if not common_images:
            raise ValueError("No matching image files found in both folders")

        print(f"Found {len(common_images)} matching image files")

        aggregate_metrics = {key: [] for key in METRIC_KEYS}

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.process_image_pair, common_images))

        valid_results = 0
        for result in results:
            if result:
                valid_results += 1
                for metric, value in result.items():
                    aggregate_metrics[metric].append(value)

        print(f"Successfully processed {valid_results} image pairs")

        mean_metrics = {
            metric: np.mean(values) if values else None
            for metric, values in aggregate_metrics.items()
        }

        try:
            mean_metrics['FID'] = self.calculate_fid()
        except Exception as e:
            print(f"Error calculating FID score: {e}")
            mean_metrics['FID'] = None

        return mean_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare images in two folders using various metrics')
    parser.add_argument('--ref_folder', type=str, help='Path to ref folder containing images')
    parser.add_argument('--target_folder', type=str, help='Path to target folder containing images')
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    comparator = ImageComparator(args.ref_folder, args.target_folder)
    metrics = comparator.run_comparison(num_threads=DEFAULT_NUM_THREADS)

    print("\nMetrics Summary:")
    print("-" * 50)
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: Failed to calculate")

    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{'_'.join(args.target_folder.split('/')[-2:])}_metrics.csv"),
        index=False,
    )
