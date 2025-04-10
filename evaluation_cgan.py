import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Define the device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, z_dim, label_dim):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(10, label_dim)  # Embedding for 10 labels
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + label_dim, 128, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        label_emb = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.expand(-1, -1, 1, 1)  # Correct shape for concatenation
        z = z.view(-1, z_dim, 1, 1)
        x = torch.cat([z, label_emb], dim=1)  # Concatenate noise and label
        return self.model(x)

def load_model():
    '''
    Load the trained cGAN generator model and set to eval mode.
    '''
    # Corrected Generator instantiation with required arguments
    model = Generator(z_dim=100, label_dim=10).to(device)
    model.load_state_dict(torch.load('./cgan_weights.pt', map_location=device))
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return model


def generate_images(model, num_images, digit):
    '''
    Generate images of a specific digit using the trained model.
    '''
    z_dim = 100
    z = torch.randn(num_images, z_dim, 1, 1, device=device)  # Latent space noise
    labels = torch.full((num_images,), digit, device=device)  # Labels for digit
    with torch.no_grad():
        generated_images = model(z, labels).cpu()
    generated_images = (generated_images + 1) / 2  # Scale from [-1, 1] to [0, 1]
    return generated_images


def plot_images(generated_images, grid_size):
    '''
    Plot generated images in a grid.
    '''
    n = grid_size
    fig, axes = plt.subplots(n, n, figsize=(5, 5))
    for i in range(n):
        for j in range(n):
            image = generated_images[i * n + j].squeeze().numpy()
            axes[i, j].imshow(image, cmap='gray')
            axes[i, j].axis('off')

    # Save the generated image to a file
    plt.savefig('cgan_generated_images.png')
    plt.show()


if __name__ == "__main__":
    # Load the cGAN generator model
    model = load_model()

    # Generate images of digit 9 (or any other digit from 0 to 9)
    num_images = 25
    digit = 9  # Change to any digit between 0-9 to generate that digit
    images = generate_images(model, num_images, digit)

    # Plot the generated images
    grid_size = 5
    plot_images(images, grid_size)

    # Visualize using IPython display
    from IPython.display import Image
    Image(filename='/content/drive/MyDrive/03.BigDataAnalytics/8.Project3/cgan_generated_images.png')