---
layout: default
title: Toy Diffusion - Understanding Diffusion Models by Example
description: A visual introduction to diffusion models using a 2D Swiss Roll dataset
---

# Understanding Diffusion Models by Example

<div class="hero-section">
  <p class="lead">An introduction to diffusion probabilistic models using a simple 2D Swiss Roll dataset for understanding the core concepts before diving into more complex applications.</p>
</div>

## What Are Diffusion Models?

Diffusion models are a class of generative models that learn to reverse a gradual noising process. Think of it like learning to "undraw" random scribbles back into meaningful data.

### The Core Idea

1. **Forward Process**: Gradually add noise to real data until it becomes pure noise
2. **Reverse Process**: Learn to remove noise step by step to generate new data
3. **Training**: Teach a neural network to predict what noise was added at each step

## The Swiss Roll Dataset

We use a 2D Swiss Roll manifold because it's:
- **Visual**: Easy to see what's happening
- **Non-trivial**: Has interesting curved structure
- **Manageable**: Small enough to train quickly

<div class="image-container">
  <img src="assets/training_results_trajectory.png" alt="Swiss Roll Dataset and Training Results" class="responsive-image">
  <p class="caption">Generating a Swiss Roll at different timesteps.</p>
</div>

## The Diffusion Process in Action

### Forward Process: Adding Noise
Starting from clean Swiss Roll data, we gradually add Gaussian noise at each timestep until the data becomes indistinguishable from random noise.

### Reverse Process: Denoising
Our trained model learns to reverse this process, starting from pure noise and gradually denoising to generate new Swiss Roll samples.

<div class="image-container">
  <img src="assets/denoising_trajectory.gif" alt="Denoising Process Animation" class="responsive-image">
  <p class="caption">Watch the reverse diffusion process: from noise to structured data</p>
</div>

### Step-by-Step Visualization

<div class="image-container">
  <img src="assets/denoising_progression_strip.png" alt="Denoising Steps" class="responsive-image">
  <p class="caption">Each step of the denoising process, from random noise (left) to clean samples (right)</p>
</div>

## Key Concepts Demonstrated

### 1. Noise Scheduling
- **Linear Schedule**: Gradually increase noise variance from β₁ ≈ 0.0001 to βₜ ≈ 0.02
- **Why Linear**: Simple and effective for this toy example

### 2. Loss Function
- **Objective**: Learn to predict the noise that was added
- **MSE Loss**: `||ε - ε_θ(x_t, t)||²` where ε is actual noise, ε_θ is predicted noise

### 3. Network Architecture
- **Simple MLP**: 3-layer neural network with ReLU activations
- **Input**: Noisy data + timestep information
- **Output**: Predicted noise vector

## Implementation Highlights

This toy implementation follows the **DDPM (Denoising Diffusion Probabilistic Models)** framework:

### Training: Learning to Predict Noise

```python
# Core training loop concept
for batch in dataloader:
    t = random_timesteps()           # Random timestep for each sample
    noise = torch.randn_like(batch)  # Sample noise
    noisy_data = add_noise(batch, noise, t)  # Forward diffusion

    predicted_noise = model(noisy_data, t)   # Network prediction
    loss = mse_loss(noise, predicted_noise)  # Compare with actual noise
```

### Sampling: Generating New Data

```python
# Core sampling loop concept
def generate_samples(model, n_samples):
    x = torch.randn(n_samples, 2)    # Start from pure noise

    # Reverse diffusion: gradually denoise
    for t in reversed(range(timesteps)):
        # Predict noise at this timestep
        predicted_noise = model(x, t)

        # Remove predicted noise (with some added randomness for t > 0)
        x = denoise_step(x, predicted_noise, t)

    return x  # Clean samples that look like Swiss Roll data
```

The magic happens in `denoise_step()` - it uses the learned noise prediction to take one step toward cleaner data, following the reverse diffusion process.

## Why This Matters

Understanding diffusion models through this simple example helps build intuition for:

- **Large-scale image generation** (DALL-E, Midjourney, Stable Diffusion)
- **Text-to-image synthesis**
- **Video generation**
- **3D shape modeling**
- **Audio generation**

The same principles scale up to high-dimensional data with more sophisticated architectures.

## Try It Yourself

```bash
# Clone and run the example
git clone https://github.com/wolrie/toy-diffusion
cd toy-diffusion
uv sync --extra=dev

# Train a model (takes ~30 seconds)
python scripts/train.py --config etc/quick_test.toml

# Check outputs/ directory for results
```

## What's Next?

- **Experiment** with different noise schedules
- **Try** different network architectures
- **Scale up** to higher-dimensional data
- **Explore** conditional generation
- **Read** the foundational papers in our [references](references.html)

---

<div class="footer-section">
  <p><strong>Built for learning.</strong> This implementation prioritizes clarity and understanding over performance.</p>
  <p>📚 <a href="references.html">Read the Papers</a> | 💻 <a href="https://github.com/wolrie/toy-diffusion">View Code</a></p>
</div>
