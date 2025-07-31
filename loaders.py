import torch
from torch.utils.data import  DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Resize(size=(28, 28)),
    v2.Normalize(mean=(0.5,), std=(0.5,)),
]
)

train_set = datasets.CIFAR10(root = 'data', train=True, transform=transform, download=True)
val_set = datasets.CIFAR10(root = 'data', train=False, transform=transform, download=True)


def get_loaders(batch_size : int = 32):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def visualize(images, labels):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).squeeze(), cmap='gray')
        plt.title(labels[i].item())
        plt.axis('off')
    plt.show()




"""
train_loader, val_loader = get_loaders()
print(len(train_loader))
print(len(val_loader))
images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

visualize(images, labels)
"""