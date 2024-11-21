from sklearn.metrics import roc_curve, auc
from watermarks.TrwStableDiffusion import TrwStableDiffusion
import os 
import torch
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image

def load_single_image(image_path):
    return Image.open(image_path)

def load_dataset(images_path,):
    # get ima
    images_paths = [os.path.join(images_path, f'data_{idx}.png') for idx in data['index']]
    
    def load_image_set(paths):
        with ThreadPoolExecutor(max_workers=24) as executor:
            images = list(tqdm(
                executor.map(lambda p: load_single_image(p), paths),
                total=len(paths),
                desc="Loading images"
            ))
        return images
    
    print("Loading images...")
    target_images = load_image_set(images_paths)
    
    return target_images, messages


def roc_curve(pred_fourier_latent, original__fourier_latent):
    generator = TrwStableDiffusion()
    no_w_metric, w_metric = [], []
    for i in [1,2,3,4]:
        x,y = verify(pred_fourier_latent, mask)
        no_w_metric.extend(x)
        w_metric.extend(y)
        true_label = [0]*len(no_w_metric) + [1]*len(w_metric)
        metric = no_w_metric + w_metric
        fpr, tpr, thresholds = roc_curve(true_label, metric)
        roc_auc = auc(fpr, tpr)
        
        
