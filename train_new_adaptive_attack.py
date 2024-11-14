import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm, trange
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
from datetime import datetime
import logging
# import lpips
# import wandb
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms

def load_single_image(image_path, transform):
    img = Image.open(image_path)
    return transform(img)

def load_attack_dataset(path, num_workers=24):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512), antialias=True)
    ])
    
    # Load CSV file
    csv_file = os.path.join(path, 'messages.csv')
    data = pd.read_csv(csv_file)
    
    # Prepare image paths
    no_watermark_paths = [os.path.join(path, 'no_watermark', f'data_{idx}.png') for idx in data['index']]
    watermark_paths = [os.path.join(path, 'watermark', f'data_{idx}.png') for idx in data['index']]
    inverse_watermark_paths = [os.path.join(path, 'inverse_watermark', f'data_{idx}.png') for idx in data['index']]
    
    # Parallel loading function
    def load_image_set(paths):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            images = list(tqdm(
                executor.map(lambda p: load_single_image(p, transform), paths),
                total=len(paths),
                desc="Loading images"
            ))
        return torch.stack(images)
    
    print("Loading no watermark images...")
    no_watermark = load_image_set(no_watermark_paths)
    
    print("Loading watermark images...")
    watermark_images = load_image_set(watermark_paths)
    
    print("Loading inverse watermark images...")
    inverse_watermark_images = load_image_set(inverse_watermark_paths)
    
    messages = torch.tensor([eval(m) for m in data['message']])
    
    return no_watermark, watermark_images, inverse_watermark_images, messages


def create_run_directory():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('training_runs', f'run_{timestamp}')
    
    # Create subdirectories with progress bar
    subdirs = ['models', 'plots', 'logs', 'samples']
    for subdir in tqdm(subdirs, desc="Creating directories"):
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    
    return run_dir

def save_sample_images(x_wm, recon_x, epoch, run_dir):
    samples_dir = os.path.join(run_dir, 'samples')
    
    with tqdm(total=3, desc=f"Saving samples for epoch {epoch}") as pbar:
        # Convert tensors to CPU and denormalize
        x_wm = (x_wm.cpu().detach() + 1.0) / 2.0
        recon_x = (recon_x.cpu().detach() + 1.0) / 2.0
        pbar.update(1)

        # Create comparison plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Plot watermarked and reconstructed images
        axes[0].imshow(torch.cat([x_wm[i] for i in range(min(4, x_wm.size(0)))], dim=2).permute(1, 2, 0))
        axes[0].set_title('Watermarked Images')
        axes[0].axis('off')
        
        axes[1].imshow(torch.cat([recon_x[i] for i in range(min(4, recon_x.size(0)))], dim=2).permute(1, 2, 0))
        axes[1].set_title('Reconstructed Images')
        axes[1].axis('off')
        pbar.update(1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f'comparison_epoch_{epoch:03d}.png'))
        plt.close()
        pbar.update(1)
        
        
def train_new_adaptive_attack(batch_size=4, num_epochs=10, learning_rate=1e-5, alpha=1, beta=1, 
                            cache_dir='cache/attack_dataset_rivagan', mode='inverse_watermark'):
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create directory for this training run
    run_dir = create_run_directory()
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load attack dataset
    logger.info("Loading attack dataset...")
    no_watermark, watermark_images, inverse_watermark_images, messages = load_attack_dataset(cache_dir)
    logger.info("Attack dataset loaded.")
    if mode == 'inverse_watermark':
        target_images = inverse_watermark_images
    elif mode == 'no_watermark':
        target_images = no_watermark
        
    # Normalize images 
    watermark_images = watermark_images.float() * 2 - 1.0
    target_images = target_images.float() * 2 - 1.0
    
    # Create DataLoader
    dataset = TensorDataset(watermark_images, target_images)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )
    
    # Initialize model with gradient checkpointing
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        subfolder="vae",
        torch_dtype=torch.float32
    ).to(device)
    vae.enable_gradient_checkpointing()
    
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    
    # Initialize W&B
    # wandb.init(project='new_adaptive-attack', name='new_adaptive-attack')
    # wandb.watch(vae)
    
    # Training loop

    for epoch in trange(num_epochs, desc="Training"):
        vae.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", 
                        leave=False, position=1)    
        for wm_batch, target_images in train_pbar:  
            target_images = target_images.to(device)
            wm_batch = wm_batch.to(device)
            optimizer.zero_grad()
            latents = vae.encode(wm_batch).latent_dist.sample()
            recon_x = vae.decode(latents).sample
            # loss = alpha * lpips_loss(recon_x, inv_wm_batch).mean() +  beta * mse_loss(recon_x, inv_wm_batch)
            loss = mse_loss(recon_x, target_images)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * wm_batch.size(0) 
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation loop
        vae.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", 
                        leave=False, position=1)
        
        with torch.no_grad():
            for wm_batch, target_images in val_pbar:  
                wm_batch = wm_batch.to(device)
                target_images = target_images.to(device)
                latents = vae.encode(wm_batch).latent_dist.sample()
                recon_x = vae.decode(latents).sample
                
                # loss = alpha *lpips_loss(recon_x, wm_batch).mean() +  beta * mse_loss(recon_x, wm_batch)
                loss = mse_loss(recon_x, wm_batch)
                val_loss += loss.item() * wm_batch.size(0)
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})


            save_sample_images(wm_batch, recon_x, epoch, run_dir)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)


        # Log to W&B
        # wandb.log({'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
        # wandb.log({'epoch': epoch+1})
        
        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(vae.state_dict(), os.path.join(run_dir, 'models', 'best_model.pth'))
            
        # Log to console
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
    logger.info(f"Training complete. Best model saved at epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    logger.info(f"Training outputs saved to: {run_dir}")
    return train_losses, val_losses

if __name__ == "__main__":
    train_new_adaptive_attack(batch_size=4,num_epochs=10, learning_rate=1e-5, alpha=1, beta=1, cache_dir='cache/attack_dataset_rivagan', mode='inverse_watermark')
