# Master's Thesis Repository: Generative AI Models for MRI Synthesis

This repository contains the complete source code, experimental pipelines, and final documentation for my Master's Thesis. The project focuses on utilizing and evaluating advanced deep learning architectures for 3D Medical Image Translation (e.g., T1-weighted to T2-weighted/FLAIR MRI synthesis) and assessing the clinical viability of the synthesized images via downstream tasks.

---

## Repository Structure & Code Details

### 1. `PTNet3D/`
Contains the source code for the **PTNet3D** architecture.
- **Functionality:** PTNet3D (Pyramid Transformer Network 3D) is designed for high-resolution volumetric image synthesis. It leverages multi-scale attention mechanisms to capture long-range structural dependencies in 3D MRI scans, outperforming traditional CNNs in maintaining global anatomical consistency. The code includes custom 3D patch-based dataloaders and grid-aggregators to handle large medical volumes without exceeding GPU memory limits.
- **Note:** This folder is cloned directly from the original authors' repository. It contains the original README and codebase.
- **Reference:** [PTNet3D: A 3D High-Resolution Longitudinal Infant Brain MRI Synthesizer Based on Transformers](https://doi.org/10.1109/TMI.2022.3174827).

### 2. `SFCN/`
Contains the implementation of the **SFCN** (Simple Fully Convolutional Network).
- **Functionality:** A lightweight, VGG-inspired fully convolutional network tailored for 3D medical imaging. It uses relatively few parameters (~3M) while achieving state-of-the-art results. In this thesis, the SFCN acts as the clinical evaluator: we train it on real MRIs and test it on synthesized MRIs to verify if the generative models (Ea-GAN, PTNet3D, Pix2Pix) preserve the crucial anatomical biomarkers required for accurate medical diagnosis.
- **Note:** This folder is cloned directly from the original authors' repository. It contains the original README and implementation details.
- **Reference:** [Accurate brain age prediction with lightweight deep neural networks (Peng et al.)](https://doi.org/10.1016/j.media.2020.101871).

### 3. `notebooks/`
A collection of standalone Jupyter Notebooks used for prototyping, training, and testing the Generative Adversarial Networks (GANs) and downstream classification models.
- **Source Note:** The implementations of **Ea-GAN** and **Pix2Pix** inside these notebooks are adapted from the [by-lab/Ea-GANs GitHub repository](https://github.com/by-lab/Ea-GANs.git).

#### Generative Models (Translation Pipelines)
- **`eagan.ipynb`**: Implementation of the Edge-aware GAN (Ea-GAN) for 3D MRI translation. 
  - **Architecture Details:** Features a dedicated 3D Sobel operator layer to extract high-frequency structural edges. 
  - **Inputs:** The Generator accepts a single modality (`real_A`), while the PatchGAN Discriminator evaluates a triplet of concatenated tensors `(real_A, real_B, edge_B)` for real samples, and `(real_A, fake_B, edge_fake_B)` for generated samples. This encourages the network to preserve fine edge details natively.
- **`pix2pix.ipynb`**: Baseline implementation of the Pix2Pix 3D framework.
  - **Architecture Details:** Utilizes a standard 3D U-Net Generator with skip connections and a classic 3D PatchGAN Discriminator. This serves as the benchmark to measure the perceptual improvements brought by Ea-GAN and PTNet3D.
- **`ptnet.ipynb`**: Experimental notebook dedicated to the training loop, loss logging, and parameter tuning of the PTNet3D model, utilizing `TorchIO` for advanced spatial augmentations.

#### Downstream Clinical Tasks
- **`Age Prediction.ipynb`**: Leverages the SFCN architecture to regress the chronological age of patients based on their brain MRIs. Used to calculate the "brain age delta" and compare the predictive accuracy when swapping real T2/FLAIR scans with synthetic ones.
- **`FCD classification.ipynb`**: Binary classification pipeline deploying the SFCN to distinguish healthy controls from subjects affected by Focal Cortical Dysplasia (FCD). Evaluates the diagnostic reliability of the generated synthetic volumes.

#### Evaluation Metrics
- **`testing from saved images.ipynb`**: A comprehensive evaluation suite. It loads batches of pre-saved NIfTI (`.nii.gz`) generated volumes and computes quantitative clinical and perceptual metrics against the ground truth, including:
  - **MAE** (Mean Absolute Error) & **PSNR** (Peak Signal-to-Noise Ratio)
  - **SSIM** (Structural Similarity Index) & **NCC** (Normalized Cross-Correlation)
  - **LPIPS** (Learned Perceptual Image Patch Similarity) & **FSIM** (Feature Similarity Index)

### 4. `latex thesis/`
Contains the final thesis manuscript and formatting assets.
- **`Thesis.tex`**: The main LaTeX source file of the thesis.
- **`bibliography.bib`**: Reference list and citations.
- **`Images/`**: Contains all figures, network architecture diagrams, and clinical plots used in the document.
- **`Configuration_files/`**: Contains LaTeX formatting, title page, and styling configurations.

---

## Setup and Requirements
The deep learning models and scripts are implemented in **PyTorch** and utilize **TorchIO** for efficient medical image handling (loading, queuing, and augmenting 3D patches). 
Please ensure all dependencies are installed before running the notebooks or training scripts.

**Hardware Note:** The Jupyter notebooks in the `notebooks/` folder are specifically configured to run on Kaggle environments, leveraging the available **NVIDIA Tesla T4 GPU equipped with 16 GB of dedicated VRAM**. Due to the memory-intensive nature of 3D convolutions, ensure your local environment matches or exceeds these specifications for optimal training performance.
