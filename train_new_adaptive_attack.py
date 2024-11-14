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
from torchvision import transforms
 
def load_attack_dataset(path):

    # Load CSV file
    csv_file = os.path.join(path, 'messages.csv')
    data = pd.read_csv(csv_file)

    # Set paths for image folders
    no_watermark_dir = os.path.join(path, 'no_watermark')
    watermark_dir = os.path.join(path, 'watermark')
    inverse_watermark_dir = os.path.join(path, 'inverse_watermark')

    no_watermark_images = []
    watermark_images = []
    inverse_watermark_images = []

    for _, row in data.iterrows():
        index = row['index']

        # Load images
        no_watermark = Image.open(os.path.join(no_watermark_dir, f'data_{index}.png'))
        watermark_img = Image.open(os.path.join(watermark_dir, f'data_{index}.png'))
        inverse_watermark_img = Image.open(os.path.join(inverse_watermark_dir, f'data_{index}.png'))

        no_watermark_images.append(no_watermark)
        watermark_images.append(watermark_img)
        inverse_watermark_images.append(inverse_watermark_img)

    no_watermark = torch.stack([transforms.ToTensor()(img) for img in no_watermark_images])
    watermark_images = torch.stack([transforms.ToTensor()(img) for img in watermark_images])
    inverse_watermark_images = torch.stack([transforms.ToTensor()(img) for img in inverse_watermark_images])
    messages = torch.tensor(list(map(lambda x: eval(x), data['message'])))
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
        
        
def train_new_adaptive_attack(batch_size=4,num_epochs=10, learning_rate=1e-5, alpha=1, beta=1, cache_dir='cache/attack_dataset_rivagan', mode='inverse_watermark'):
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create directory for this training run
    run_dir = create_run_directory()
    logger.info(f"Training outputs will be saved to: {run_dir}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")


    # Load attack dataset
    logger.info("Loading attack dataset...")
    no_watermark, watermark_images, inverse_watermark_images, messages = load_attack_dataset(cache_dir)
    logger.info("Attack dataset loaded.")
    
    # Normalize images
    with tqdm(total=3, desc="Preparing dataset") as pbar:    
        no_watermark = no_watermark.to(device).float() * 2 - 1.0
        watermark_images = watermark_images.to(device).float() * 2 - 1.0
        inverse_watermark_images = inverse_watermark_images.to(device).float() * 2 - 1.0
        pbar.update(1)
        if watermark_images.size(-1) != 768 or watermark_images.size(-2) != 768:
            watermark_images = nn.functional.interpolate(watermark_images, (768, 768), mode='bilinear')
            inverse_watermark_images = nn.functional.interpolate(inverse_watermark_images, (768, 768), mode='bilinear')
            no_watermark = nn.functional.interpolate(no_watermark, (768, 768), mode='bilinear')
        pbar.update(1)
        # Create DataLoader
        dataset = TensorDataset(no_watermark, watermark_images, inverse_watermark_images)
        train_size = int(0.75 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        pbar.update(1)
        
        with tqdm(total=1, desc="Initializing model") as pbar:
            # Initialize model
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                subfolder="vae",
                torch_dtype=torch.float32
            ).to(device)
            pbar.update(1)
            optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
            # lpips_loss = lpips.LPIPS(net='alex').to(device)
            pbar.update(1)
            mse_loss = nn.MSELoss()  
            pbar.update(1)

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
            for orig_batch, wm_batch, inv_wm_batch in train_loader:
                    
                optimizer.zero_grad()
                latents = vae.encode(wm_batch).latent_dist.sample()
                recon_x = vae.decode(latents).sample
                if mode == 'inverse_watermark':
                    # loss = alpha * lpips_loss(recon_x, inv_wm_batch).mean() +  beta * mse_loss(recon_x, inv_wm_batch)
                    loss = mse_loss(recon_x, inv_wm_batch)
                elif mode == 'no_watermark':
                    # loss = alpha * lpips_loss(recon_x, orig_batch).mean() + beta * mse_loss(recon_x, orig_batch)
                    loss = mse_loss(recon_x, orig_batch)
                    
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
            for orig_batch, wm_batch, inv_wm_batch in val_loader:
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
