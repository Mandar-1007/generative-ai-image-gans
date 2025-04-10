# evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
z_dim = 100  # Added z_dim

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, z_dim, 1, 1)
        return self.model(x)

def load_model():
    model = Generator().to(device)
    model.load_state_dict(torch.load('./generator_weights.pt', map_location=device))
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return model

def generate_images(model, num_images, seed=42):
    torch.manual_seed(seed)  # 
    z = torch.randn(num_images, z_dim, 1, 1, device=device)
    with torch.no_grad():
        generated_images = model(z).cpu()
    generated_images = (generated_images + 1) / 2
    return generated_images

def plot_selected_images(images, indices, save_path='selected_grid.png'):
    selected_images = images[indices]
    fig, axes = plt.subplots(5, 5, figsize=(5, 5))
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            axes[i, j].imshow(selected_images[idx][0], cmap='gray')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_all_grid(images, filename="grid_100.png"):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            idx = i * 10 + j
            axes[i, j].imshow(images[idx][0], cmap='gray')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    model = load_model()
    images = generate_images(model, 100, seed=42)

    # Show 100 images in a grid
    #plot_all_grid(images, filename="grid_seed42_100.png")

    # Manually updated selected indices (you should change these after reviewing grid_100.png)
    selected_indices = [24, 65,  0, 93, 4, 3,  87, 77, 18, 29, 11, 7, 84, 94, 10, 44, 49, 19, 66, 67, 77, 90, 58, 79, 56]

    # Final 5x5 output grid
    # plot_selected_images(images, selected_indices)
    #plot_selected_images(images, selected_indices, save_path='600_epochs_grid1.png')

    # Generate an alternate 100 samples using a different seed (to look for digit 6)
    images_100_seed99 = generate_images(model, 100, seed=99)
    #plot_all_grid(images_100_seed99, filename="grid_seed99_100.png")

    selected_indices2 = [60, 50, 25, 91, 6, 50, 22, 5, 50, 72, 34, 19, 14, 52, 28, 88, 2, 27, 31, 52, 41, 6, 49, 1, 62]

    # Final 5x5 output grid
    plot_selected_images(images_100_seed99, selected_indices2, save_path='600_epochs_grid2.png')