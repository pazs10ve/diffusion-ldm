from vae import VAE
from loaders import get_loaders
from utils import save_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm


def train_vae(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_idx, (images, _) in enumerate(tqdm(train_loader, leave=False)):
            images = images.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var, _ = model(images)
            loss = criterion(recon_batch, images)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        val_loss = evaluate(val_loader, model, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader)}, Validation Loss : {val_loss/len(val_loader)}')

        ckp_path = f'saved_models/model_ckp_{epoch+1}.pt'
        save_model(model, ckp_path)
            

def evaluate(data_loader, model, criterion, device):
    val_loss = 0.0
    model.eval()

    for batch_idx, (images, _) in enumerate(tqdm(data_loader, leave=False)):
        images = images.to(device)
        with torch.no_grad():
            recon_batch, mu, log_var, _ = model(images)
            loss = criterion(recon_batch, images)
            val_loss += loss.item()

    return val_loss


batch_size = 16
num_epochs = 1000
learning_rate = 2e-6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_loader, val_loader = get_loaders()
print(f'The size of training set is {len(train_loader)}')
print(f'The size of validation set is {len(val_loader)}')

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()


train_vae(train_loader, val_loader, model, criterion, optimizer, num_epochs, device)
