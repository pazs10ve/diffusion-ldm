from vae import VAE
from loaders import get_loaders
from utils import save_model
from model import UNet, LatentDiffusionModel


import torch
import torch.optim as optim
from tqdm import tqdm
import os


def train_ldm(train_loader, val_loader, optimizer, num_epochs, device, vae_checkpoint_path):
    
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(vae_checkpoint_path, map_location=device))
    vae.eval()
    
    unet = UNet().to(device)
    
    ldm = LatentDiffusionModel(vae, unet).to(device)
    
    optimizer = optim.Adam(ldm.unet.parameters(), lr=2e-4)
    
    os.makedirs('saved_ldm_models', exist_ok=True)
    
    for epoch in range(num_epochs):
        ldm.train()
        train_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            images = images.to(device)
            optimizer.zero_grad()
            
            loss = ldm(images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        val_loss = evaluate_ldm(val_loader, ldm, device)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Training Loss: {avg_train_loss:.6f}')
        print(f'  Validation Loss: {avg_val_loss:.6f}')
        
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'unet_state_dict': ldm.unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            torch.save(checkpoint, f'saved_ldm_models/ldm_checkpoint_epoch_{epoch+1}.pt')
            print(f'Checkpoint saved: ldm_checkpoint_epoch_{epoch+1}.pt')


def evaluate_ldm(data_loader, model, device):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc='Validating', leave=False)):
            images = images.to(device)
            loss = model(images)
            val_loss += loss.item()
    
    return val_loss


def generate_samples(ldm_checkpoint_path, vae_checkpoint_path, num_samples=8, device='cuda'):
    
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(vae_checkpoint_path, map_location=device))
    vae.eval()
    
    unet = UNet().to(device)
    
    ldm = LatentDiffusionModel(vae, unet).to(device)
    
    checkpoint = torch.load(ldm_checkpoint_path, map_location=device)
    ldm.unet.load_state_dict(checkpoint['unet_state_dict'])
    ldm.eval()
    
    with torch.no_grad():
        generated_images = ldm.sample(batch_size=num_samples, device=device)
    
    return generated_images



if __name__ == "__main__":
    batch_size = 16
    num_epochs = 100
    learning_rate = 2e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_loaders()
    print(f'Training set size: {len(train_loader)}')
    print(f'Validation set size: {len(val_loader)}')

    vae_checkpoint_path = 'saved_models/model_ckp_1000.pt' 
    
    if not os.path.exists(vae_checkpoint_path):
        print(f"VAE checkpoint not found at {vae_checkpoint_path}")
        print("Please train VAE first or adjust the path.")
        exit(1)
    
    optimizer = None
    
    print("Starting Latent Diffusion Model training...")
    train_ldm(train_loader, val_loader, optimizer, num_epochs, device, vae_checkpoint_path)
    
    print("Training completed!")
    



   
  