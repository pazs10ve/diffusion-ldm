from vae import VAE
from utils import load_model
from main import generate_samples

import os
import torch
import matplotlib.pyplot as plt


x = torch.randn(1, 56)
model = VAE()
model = load_model(model, 'saved_models/model_ckp_300.pt')
out = model.decode(x)
print(out.shape)

plt.imshow(out[0].permute(1, 2, 0).squeeze().detach().numpy(), cmap='gray')
plt.show()


vae_checkpoint_path = 'saved_models/model_ckp_1000.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


latest_checkpoint = 'saved_ldm_models/ldm_checkpoint_epoch_100.pt'
if os.path.exists(latest_checkpoint):
    samples = generate_samples(latest_checkpoint, vae_checkpoint_path, num_samples=8, device=device)
    torch.save(samples, 'generated_samples.pt')