# Generative AI for Image Synthesis using GANs, cGANs, and CycleGANs

This project explores image generation using Generative Adversarial Networks (GANs), implemented and tested across three major variants: Vanilla GAN, Conditional GAN (cGAN), and CycleGAN. The project is structured into three parts:

**Part 1 – Vanilla GAN (MNIST Digit Generation):**
A basic GAN is trained on the MNIST dataset to generate handwritten digit images from random noise. Extensive experimentation and tuning were conducted across 600 epochs to improve digit clarity and diversity.

**Part 2 – Conditional GAN (Class-Controlled MNIST Generation):**
Extends the vanilla GAN by conditioning the generation process on digit labels (0–9), enabling controlled generation. The model generates images of a specific digit based on the input label, showcasing the power of supervised conditioning.

**Part 3 – CycleGAN (Unpaired Image-to-Image Translation – Apple ↔ Orange):**
Implements CycleGAN to perform unpaired translation between apples and oranges. The model learns two mappings: Apple→Orange and Orange→Apple, using cycle consistency loss and identity loss. The dataset is unpaired and real-world, highlighting the model's ability to generalize.

Each part includes:
- Complete training and testing pipeline in PyTorch
- Model checkpoints and loss tracking
- Experimentation insights (optimizers, seeds, label smoothing, etc.)
- Final outputs as image grids and translated samples

# Dataset Used

**Part 1 & Part 2 – MNIST Handwritten Digit Dataset**
1) Source:
torchvision.datasets.MNIST
2) Description:
A classic benchmark dataset containing 60,000 training and 10,000 test images of grayscale handwritten digits (0–9), each of size 28×28 pixels.
3) Usage:
- Vanilla GAN learns to generate digits from noise without label information.
- Conditional GAN is trained with digit labels (0–9) to enable class-controlled generation

**Part 3 – Apple ↔ Orange Translation Dataset**
1) Source:
https://www.kaggle.com/datasets/balraj98/apple2orange-dataset
2) Description:
A CycleGAN-specific unpaired dataset consisting of images of apples (domain A) and oranges (domain B) for both training and testing.
3) Structure:

![image](https://github.com/user-attachments/assets/45793575-5b4e-4ee1-90ab-81c7e7738b5e)

4) Usage:
Enables unpaired image-to-image translation between two visual domains without needing aligned examples.

# Experiments

This project is divided into three parts—Vanilla GAN, Conditional GAN, and CycleGAN—each exploring different generative modeling techniques and their practical implementations. Below is a summary of experiments and progress across all parts:

**🔹Part 1 – Vanilla GAN on MNIST**

Objective: Train a basic GAN to generate handwritten digits using only noise vectors (no labels).

**Key Experiments:**
1) Initial Training (Epochs 0–300):
- Optimizer: RMSprop
- Label Smoothing (real labels set to 0.8 + ε)
- Gradient Penalty added for training stability
- Generator weights saved every 50 epochs
- Initial samples lacked clarity and realism

2) Resume Training (Epochs 300–400):
- Switched to Adam optimizer (lr=0.0001, betas=(0.5, 0.999))
- Removed label smoothing
- Fixed random seed = 42 for reproducibility of generated images
- Output grid significantly improved (0–9 digits identifiable, except 6)

3) Continued Training (Epochs 400–600):
- Continued with Adam optimizer and fixed seed
- Improved loss curves and stability
- Generated 10×10 grid with seed 42 and alternative grid with seed 99
- Selected top 25 clear digits manually from the second grid for final 5×5

**Visual Outputs:**
1) Loss plot:
- vanilla_gan_loss_plot.png

2) Final generated grids:
- grid_seed42_100.png, 600_epochs_grid1.png
- grid_seed99_100.png, 600_epochs_grid2.png (includes digit 6)

3) Model files:
generator_weights.pt, generator.pt

**🔹 Part 2 – Conditional GAN on MNIST**

Objective: Extend GAN to generate digits conditioned on specific class labels (0–9).

**Key Experiments:**
- Used embedding layer for digit labels
- Concatenated noise and label embeddings in the Generator
- Embedded labels as extra channel in the Discriminator
- Trained for 150 epochs using RMSprop
- Applied label smoothing for discriminator training
- Generated 5×5 image grids for digits 0 to 9 with clear label control

**Visual Outputs:**
- 25 samples of specific digit (e.g., digit 9): cgan_generated_images.png

**Model files:**
- Generator weights: cgan_weights.pt
- Full model: cgan_generator.pt

**🔹 Part 3 – CycleGAN: Apple ↔ Orange Translation**

Objective: Perform unpaired image-to-image translation between apples and oranges using CycleGAN.

**Key Experiments:**
- Built ResNet-based generators with dropout for regularization
- Used PatchGAN discriminators with 70×70 receptive fields
- Included Cycle-consistency loss and Identity loss with weights:
- λ_cycle = 10, λ_identity = 5
- Used replay buffer for stability
- Trained for 250 epochs on apple2orange dataset
- Generated bidirectional translations using random test images from testA and testB
- Saved translated images after each inference

**Visual Outputs:**
1) Translated image samples:
- Apple → Orange: apple_to_orange_1.png, apple_to_orange_2.png, ...
- Orange → Apple: orange_to_apple_1.png, orange_to_apple_2.png, ...

# Techniques and Special Skills Used
To improve the stability, quality, and control of generated images across all three parts of the project, several advanced GAN training techniques and best practices were incorporated. Each part leveraged different sets of strategies as outlined below:

**🔹 Part 1 – Vanilla GAN**
1) Input Normalization: Images scaled to −1,1 with Tanh activation in the final generator layer.
2) Noise Distribution: Sampled latent vectors z from a standard normal distribution (N(0,1)), ensuring diverse and continuous representations.
3) Soft and Noisy Labels:
  - Applied label smoothing with real labels = 0.8 + ε, and fake = 0.0 + ε during initial training.
  - Added random noise to labels for stability.
4) Gradient Penalty: Introduced gradient penalty to the discriminator to improve Lipschitz continuity and mitigate overfitting.
5) Optimizer Tuning:
  - Initially used RMSprop; later switched to Adam with momentum (betas=(0.5, 0.999)), which significantly stabilized training.
6) Seed Fixation: Used fixed random seeds (42 and 99) to ensure reproducibility and facilitate meaningful grid comparison.
7) Manual Curation: Manually selected 25 clearly identifiable digits from 100 samples to form visually interpretable grids.
8) Training Resumption & Checkpointing: Implemented mechanisms to resume training from any epoch, ensuring efficient experimentation without starting from scratch

**🔹 Part 2 – Conditional GAN (cGAN)**
1) Label Conditioning via Embedding: Used an embedding layer to transform integer digit labels into a learnable representation space.
2) Concatenation Strategy:
  - Concatenated latent vector z with label embedding in the generator.
  - Added label embeddings as a separate channel in the discriminator input.
3) Input Normalization: All images normalized to −1,1 with Tanh as the generator's final activation.
4) RMSprop Optimization: Used consistent RMSprop optimizers for both generator and discriminator.
5) Class-Controlled Generation: Enabled controlled image generation by passing desired digit labels to the generator at inference time.
6) Evaluation Routine: Developed a reusable and customizable evaluation module for generating 5×5 digit grids of specific classes.

**🔹 Part 3 – CycleGAN: Apple ↔ Orange**
1) ResNet-Based Generator: Implemented generators with multiple residual blocks and dropout to improve robustness and generalization.
2) PatchGAN Discriminator: Used discriminators with 70×70 receptive fields for high-frequency detail preservation.
3) Cycle Consistency Loss: Enforced image structure preservation using L1 loss between original and reconstructed images.
4) Identity Loss: Added identity mapping loss to encourage color consistency during domain translation.
5) Replay Buffer: Incorporated a replay buffer for discriminator training, improving model stability and preventing mode collapse.
6) Dropout During Training: Applied dropout in the residual blocks of the generator during training to introduce stochasticity and improve generalization.
7) Test-Time Inference Module: Enabled flexible translation of any image via generate_and_plot(...), supporting custom test image evaluation.

# Visual Results

This section showcases the progression and final outputs of all three parts of the project — demonstrating improved image clarity, class control, and unpaired translation over time.

**🔹 Part 1 – Vanilla GAN (MNIST Digits)**
1) Progress over training:
The quality of generated digits improved significantly as training progressed. Below are sample outputs after 300, 400, and 600 epochs.

After 300 Epochs:

![300 epochs](https://github.com/user-attachments/assets/22680efd-639f-4ce6-ba4d-0cbb44254ecc)

After 400 Epochs:

![400 epochs](https://github.com/user-attachments/assets/0394190c-1cbd-47be-872a-ee078c05ed7d)

10 X 10 Grid (Seed 42):

![grid_seed42_100](https://github.com/user-attachments/assets/176687d0-53de-4584-87ed-3d3f5599ed83)

10 X 10 Grid (Seed 99):

![grid_seed99_100](https://github.com/user-attachments/assets/adabc251-16be-4d5f-a2ad-02c29e8092c6)

Final 5×5 Grid (Manually Selected):
25 best digits selected for clarity.

![600_epochs_grid2](https://github.com/user-attachments/assets/1a87d0ba-ceb2-4f87-9de5-4d9452b56f3b)


**🔹 Part 2 – Conditional GAN (cGAN)**

Final Grid Output Example (Digit 9):

![cgan_generated_images](https://github.com/user-attachments/assets/3795008e-922b-4973-90a5-f6e279676648)


**🔹 Part 3 – CycleGAN (Apple ↔ Orange)**

Apple → Orange Translation:


1) Image 1:

![apple_to_orange_3](https://github.com/user-attachments/assets/9f49def9-028e-443c-b539-9e18092f3cb9)

2) Image 2:

![apple_to_orange_1](https://github.com/user-attachments/assets/11781f52-8730-42e3-b50f-93dffebd5983)

4) Image 3:

![apple_to_orange_3](https://github.com/user-attachments/assets/c3f97446-906f-45f7-bcd3-38d0f7d189e6)


Orange → Apple Translation:

1) Image 1:

![orange_to_apple_1 (1)](https://github.com/user-attachments/assets/80864d0d-e0f7-420d-9c51-79ee2cc7aba3)

2) Image 2:

![orange_to_apple_1](https://github.com/user-attachments/assets/82efe292-0383-4011-b37d-4fb6769de2f5)

3) Image 3:

![orange_to_apple_2](https://github.com/user-attachments/assets/3be2463a-bb45-42d1-b932-5acdec162955)

# Dependencies

This project was developed using Python 3.10+ and the following core libraries:

**Python Libraries:**
1) torch - Deep learning framework for building GANs
2) torchvision - Datasets and image transforms
3) matplotlib - Plotting loss curves and image grids
4) numpy - Array operations and numeric utilities
5) Pillow - Image loading for CycleGAN
6) IPython.display - Displaying images in Jupyter Notebooks
7) glob & os - File system navigation and file access
8) random - Random sampling for test images
9) scikit-learn - For advanced sampling/metrics (optional)

# Environment

- Google Colab was used for training and testing all models (recommended)
- All experiments were run using GPU acceleration where available

*Note: If running locally, ensure you have a CUDA-compatible GPU and the proper PyTorch installation*

# Conclusion

This project demonstrates the practical implementation of Generative Adversarial Networks (GANs) across three major paradigms:
- Vanilla GAN – for learning to generate handwritten digits in an unsupervised setting.
- Conditional GAN (cGAN) – to control generation using class labels, allowing precise digit synthesis.
- CycleGAN – for unpaired image-to-image translation between two real-world domains: apples and oranges.

Through progressive training, careful loss function tuning, and GAN best practices (label smoothing, batch normalization, dropout, gradient penalties), we successfully enhanced the clarity, stability, and diversity of generated outputs. This project not only explores core GAN architectures but also emphasizes hands-on experimentation, training stabilization techniques, and qualitative evaluation via side-by-side image comparisons.
