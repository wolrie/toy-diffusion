# References

This document lists the key papers and resources that informed the implementation of this toy diffusion model.

### 1. Deep Unsupervised Learning using Nonequilibrium Thermodynamics
[PMLR](https://proceedings.mlr.press/v37/sohl-dickstein15.html) | [ArXiv](https://arxiv.org/abs/1503.03585) | [GitHub](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models) | [Papers with Code](https://paperswithcode.com/paper/deep-unsupervised-learning-using)

**Authors:** Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, Surya Ganguli
**Conference:** PMLR 2015 (Proceedings of Machine Learning Research)
**Pages:** 2256-2265

**Summary:** This seminal paper introduces the foundational concept of diffusion models. The authors develop an approach inspired by non-equilibrium statistical physics, where structure in data is systematically destroyed through an iterative forward diffusion process, and then a reverse diffusion process is learned to restore structure, creating a highly flexible generative model.

**Key Contributions:**
- Introduction of the forward and reverse diffusion process framework
- Theoretical foundation linking thermodynamics to generative modeling
- Tractable learning and sampling algorithms for deep generative models

### 2. Denoising Diffusion Probabilistic Models
[NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) | [ArXiv](https://arxiv.org/abs/2006.11239) | [GitHub](https://github.com/hojonathanho/diffusion)

**Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel
**Conference:** NeurIPS 2020 (33rd Conference on Neural Information Processing Systems)
**Article:** No. 574, Pages 6840-6851

**Summary:** This paper presents the DDPM (Denoising Diffusion Probabilistic Models) algorithm, which significantly improves upon the original diffusion model framework. It introduces the key insight of parameterizing the reverse process as a denoising problem and provides practical training algorithms that achieve state-of-the-art results.

**Key Contributions:**
- Simplified training objective based on denoising score matching
- Improved sampling quality and stability
- Connection between diffusion models and denoising autoencoders
- Practical implementation details for training deep diffusion models
