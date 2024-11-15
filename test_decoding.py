from watermarks.Rivagan import Rivagan
from watermarks.StegaStamp import StegaStamp
from PIL import Image
import os
import torch
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

def load_single_image(image_path):
    img = Image.open(image_path)
    return img

def load_attack_dataset(images_path, csv_path, num_workers=24):

    
    csv_file = os.path.join(csv_path, 'messages.csv')
    data = pd.read_csv(csv_file)
    
    # Prepare image paths
    images_paths = [os.path.join(images_path, f'data_{idx}.png') for idx in data['index']]
    
    # Parallel loading function
    def load_image_set(paths):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            images = list(tqdm(
                executor.map(lambda p: load_single_image(p), paths),
                total=len(paths),
                desc="Loading images"
            ))
        return images
    
    print("Loading images...")
    target_images = load_image_set(images_paths)
    
    messages = torch.tensor([eval(m) for m in data['message']])
    
    return target_images, messages

@torch.no_grad()
def test_decoding(images_path='cache/test_dataset_rivagan', csv_path='cache/test_dataset_rivagan', watermark_algrothim='rivagan', batch_size=16):
    if watermark_algrothim == 'rivagan':
        watermark_key = Rivagan()
    elif watermark_algrothim == 'stegastamp':
        watermark_key = StegaStamp()
    
    target_images, messages = load_attack_dataset(images_path, csv_path)
    num_batches = len(target_images) // batch_size
    decoded_messages = []
    for i in range(num_batches):
        batch_images = target_images[i*batch_size:(i+1)*batch_size]
        print(f"Batch {i+1}/{num_batches}")
        decoded_messages.append(watermark_key.decode(batch_images))
        
    decoded_messages = torch.from_numpy(np.stack(decoded_messages))
    messages = messages.view(num_batches, batch_size, -1)
    accuracy = (decoded_messages == messages).float().mean().item()
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    test_decoding(images_path='attacked/stegastamp_testset_stegastamp_no_watermark_mse',csv_path='cache/test_dataset_stegastamp', watermark_algrothim='stegastamp', batch_size=16)
