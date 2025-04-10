# training_cgan.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
num_classes = 10  # Digits 0-9
label_dim = 10  # Embedding for 10 classes


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


class Discriminator(nn.Module):
    def __init__(self, label_dim):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, 28 * 28)  # Embed to 28x28
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),  # Correct input channels: 2 (image + label)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels)  # Shape: (batch_size, 28*28)
        label_emb = label_emb.view(-1, 1, 28, 28)  # Reshape to (batch_size, 1, 28, 28)
        x = torch.cat([img, label_emb], dim=1)  # Concatenate along channel axis
        return self.model(x)


def train_cgan(epochs=150, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    G = Generator(z_dim, label_dim).to(device)
    D = Discriminator(label_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizerG = optim.RMSprop(G.parameters(), lr=0.00005)
    optimizerD = optim.RMSprop(D.parameters(), lr=0.00005)

    G.train()
    D.train()

    for epoch in range(epochs):
        for i, (real_images, labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)

            real_labels = torch.full((batch_size, 1), 0.8, device=device)
            fake_labels = torch.full((batch_size, 1), 0.0, device=device)

            # --- Train Discriminator ---
            outputs_real = D(real_images, labels)
            d_loss_real = criterion(outputs_real, real_labels)

            z = torch.randn(batch_size, z_dim, 1, 1).to(device)
            random_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_images = G(z, random_labels)
            outputs_fake = D(fake_images.detach(), random_labels)
            d_loss_fake = criterion(outputs_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            D.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # --- Train Generator ---
            z = torch.randn(batch_size, z_dim, 1, 1).to(device)
            random_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_images = G(z, random_labels)
            outputs = D(fake_images, random_labels)
            g_loss = criterion(outputs, real_labels)

            G.zero_grad()
            g_loss.backward()
            optimizerG.step()

        print(f"Epoch [{epoch + 1}/{epochs}]  Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

        # Save model every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(G.state_dict(), f'./cgan_weights_epoch_{epoch + 1}.pt')

    # Save final model
    torch.save(G.state_dict(), './cgan_weights.pt')
    return G


def plot_generated_images(model, num_images=25, digit=0):
    model.eval()
    z = torch.randn(num_images, z_dim, 1, 1).to(device)
    labels = torch.full((num_images,), digit, device=device)
    generated_images = model(z, labels).detach().cpu()
    generated_images = (generated_images + 1) / 2  # Scale to [0, 1]

    n = int(np.sqrt(num_images))
    fig, axes = plt.subplots(n, n, figsize=(5, 5))
    for i in range(n):
        for j in range(n):
            axes[i, j].imshow(generated_images[i * n + j][0], cmap='gray')
            axes[i, j].axis('off')
    plt.show()