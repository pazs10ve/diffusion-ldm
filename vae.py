import torch
import torch.nn as nn 


class VaeEncoder(nn.Module):
    def __init__(self, in_channels:int = 3, latent_dim:int = 56, image_size=28):
        super(VaeEncoder, self).__init__()
        layers = []

        channels = [in_channels, 32, 32]

        for i in range(len(channels) - 1):
            layers.extend([nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1),
                          nn.BatchNorm2d(num_features=channels[i+1]),
                          nn.LeakyReLU(0.2, inplace=True)])
        
        self.encoder = nn.Sequential(*layers)
        image_size = image_size // (2 ** (len(channels) - 1))

        self.log_var = nn.Linear(in_features=32*image_size*image_size, out_features=latent_dim)
        self.mu = nn.Linear(in_features=32*image_size*image_size, out_features=latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var
    


class VaeDecoder(nn.Module):
    def __init__(self, latent_dim:int=56, out_channels:int=3, image_size=28):
        super(VaeDecoder, self).__init__()
        final_size = 7
        self.fc = nn.Linear(latent_dim, 32*final_size*final_size)
        layers = []

        channels = [32, 32, out_channels]
        
        for i in range(len(channels) - 1):
            layers.extend([nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                          nn.BatchNorm2d(num_features=channels[i+1]),
                          nn.LeakyReLU(0.2, inplace=True)])
            

        layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*layers)   

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 32, 7, 7)
        h = self.decoder(h)
        return h
    


def reparameterize(mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return mu + eps*std



class VAE(nn.Module):
    def __init__(self, in_channels:int = 3, latent_dim:int = 56, image_size=28):
        super(VAE, self).__init__()
        self.encoder = VaeEncoder(in_channels, latent_dim, image_size)
        self.decoder = VaeDecoder(latent_dim, in_channels, image_size)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var, z
    
    def decode(self, z):
        out = self.decoder(z)
        return out
    
    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu 
    





"""
x = torch.randn(1, 3, 28, 28)
encoder = VaeEncoder()
out = encoder(x)
print(out[0].shape, out[1].shape)
"""


"""
x = torch.randn(1, 56)
decoder = VaeDecoder()
out = decoder(x)
print(out.shape)
"""


"""
x = torch.randn(1, 3, 28, 28)
vae = VAE()
out = vae(x)
print(out[0].shape, out[1].shape, out[2].shape, out[3].shape)
"""