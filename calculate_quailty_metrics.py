import os
import torch
import lpips
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_mutual_information
from torchvision import transforms
from pytorch_fid import fid_score
from concurrent.futures import ThreadPoolExecutor
import argparse
import pandas as pd

class ImageComparator:
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def __init__(self, folder1, folder2):
        self.folder1 = folder1
        self.folder2 = folder2
        self.metrics = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    @staticmethod
    def is_image_file(filename):
        """Check if a file is an image based on its extension"""
        return os.path.splitext(filename.lower())[1] in ImageComparator.ALLOWED_IMAGE_EXTENSIONS
        
    def calculate_metrics(self, img1_path, img2_path):
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Convert to numpy arrays
            img1_np = np.array(img1)
            img2_np = np.array(img2)
            
            # Convert to torch tensors for LPIPS
            img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
            img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
            
            # Calculate metrics
            metrics = {
                'LPIPS': float(self.lpips_fn(img1_tensor, img2_tensor).item()),
                'MSE': np.mean((img1_np - img2_np) ** 2),
                'PSNR': peak_signal_noise_ratio(img1_np, img2_np),
                'SSIM': structural_similarity(img1_np, img2_np, channel_axis=2),
                'NMI': normalized_mutual_information(img1_np, img2_np)
            }
            
            return metrics
        except Exception as e:
            print(f"Error processing images {img1_path} and {img2_path}: {str(e)}")
            return None
    
    def process_image_pair(self, img_name):
        img1_path = os.path.join(self.folder1, img_name)
        img2_path = os.path.join(self.folder2, img_name)
        
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            return self.calculate_metrics(img1_path, img2_path)
        return None
    
    def calculate_fid(self):
        return fid_score.calculate_fid_given_paths(
            [self.folder1, self.folder2],
            50,  # batch size
            self.device,
            2048  # dims
        )
    
    def get_image_files(self, folder):
        """Get list of image files in a folder"""
        return [f for f in os.listdir(folder) if self.is_image_file(f)]
    
    def run_comparison(self, num_threads=4):
        # Get common images between folders (only image files)
        images1 = set(self.get_image_files(self.folder1))
        images2 = set(self.get_image_files(self.folder2))
        common_images = list(images1.intersection(images2))
        
        if not common_images:
            raise ValueError("No matching image files found in both folders")
        
        print(f"Found {len(common_images)} matching image files")
        
        # Initialize aggregate metrics
        aggregate_metrics = {
            'LPIPS': [],
            'MSE': [],
            'PSNR': [],
            'SSIM': [],
            'NMI': []
        }
        
        # Process images using thread pool
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.process_image_pair, common_images))
            
        # Aggregate results
        valid_results = 0
        for result in results:
            if result:
                valid_results += 1 
                for metric, value in result.items():
                    aggregate_metrics[metric].append(value)
        
        print(f"Successfully processed {valid_results} image pairs")
        
        # Calculate means
        mean_metrics = {
            metric: np.mean(values) if values else None 
            for metric, values in aggregate_metrics.items()
        }
        
        # Calculate FID score
        try:
            mean_metrics['FID'] = self.calculate_fid()
        except Exception as e:
            print(f"Error calculating FID score: {e}")
            mean_metrics['FID'] = None
        
        return mean_metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare images in two folders using various metrics')
    parser.add_argument('--ref_folder', type=str, help='Path to ref folder containing images')
    parser.add_argument('--target_folder', type=str, help='Path to target folder containing images')
    
    args = parser.parse_args()
    
    comparator = ImageComparator(args.ref_folder, args.target_folder)
    metrics = comparator.run_comparison(num_threads=24)
    
    print("\nMetrics Summary:")
    print("-" * 50)
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: Failed to calculate")
    # save csv

    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(os.path.join('./quality', f"{'_'.join(args.target_folder.split('/')[-2:])}_metrics.csv"), index=False)

