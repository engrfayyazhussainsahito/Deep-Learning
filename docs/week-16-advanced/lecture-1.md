# Week 16: Generative AI: GANs and Diffusion Models

## Theory: Learning to Sample from High-Dimensional Distributions

Generative modeling is the task of estimating the underlying probability distribution $p_{data}(\mathbf{x})$ of a dataset and sampling new points $\mathbf{x}_{new} \sim p_{model}(\mathbf{x})$.

### 1. Generative Adversarial Networks (GANs)
GANs treat generative modeling as a two-player minimax game.
*   **The Generator ($G$):** Maps a latent noise vector $\mathbf{z} \sim \mathcal{N}(0, I)$ to a data sample $G(\mathbf{z})$.
*   **The Discriminator ($D$):** A binary classifier that outputs the probability that $\mathbf{x}$ came from the real data rather than $G$.

The objective function is:
$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$

#### The Stability Challenge
GANs are notoriously hard to train due to **Mode Collapse** (where $G$ produces only one type of output) and **Vanishing Gradients** (when $D$ becomes too perfect, $G$ receives no useful signal).

### 2. Diffusion Models: Reversing Entropy
Diffusion models (like Stable Diffusion) replace the adversarial game with a two-step stochastic process.

#### The Forward Process (Diffusion)
Gradually add Gaussian noise to an image $\mathbf{x}_0$ over $T$ steps until it becomes pure noise $\mathbf{x}_T$.
$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$

#### The Reverse Process (Denoising)
A neural network (usually a U-Net) is trained to predict the noise added at each step, effectively "undoing" the entropy.
$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$

:::tip Academic Note
The fundamental difference: GANs learn to map noise to data in one step (implicit distribution), while Diffusion models learn to iteratively refine noise into data (explicit score matching).
:::

## Modular PyTorch Implementation: Generator Architecture

We implement a professional Generator and Discriminator pair for a simple MNIST-style generation task.

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh() # Output range [-1, 1]
        )
        
    def forward(self, z):
        return self.net(z).view(z.shape[0], *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

# Setup
z_dim = 100
img_shape = (1, 28, 28)
gen = Generator(z_dim, img_shape)
disc = Discriminator(img_shape)

noise = torch.randn(1, z_dim)
fake_img = gen(noise)
validity = disc(fake_img)
print(f"Generated Image Shape: {fake_img.shape}, Validity Score: {validity.item():.4f}")
```

## Rigorous Assignment 16: Generative Dynamics

1.  **WGAN Analysis**: Research the **Wasserstein GAN (WGAN)**. Prove how using the Earth Mover's distance instead of Jensen-Shannon divergence solves the vanishing gradient problem.
2.  **Diffusion Schedulers**: Compare Linear vs Cosine noise schedules in Diffusion models. How does the schedule affect the quality of the final image?
3.  **Implementation Challenge**: Build a simple GAN to generate handwritten digits. Plot the loss of $G$ and $D$ over 50 epochs and explain any oscillations.
