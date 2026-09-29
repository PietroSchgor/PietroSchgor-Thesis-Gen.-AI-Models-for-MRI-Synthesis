# Master's Thesis Repository

This repository contains the complete source code, experimental notebooks, and final documentation for my Master's Thesis. 

## Repository Structure

### 1. `PTNet3D/`
Contains the source code for the **PTNet3D** architecture (3D High-Resolution Longitudinal Infant Brain MRI Synthesizer Based on Transformers). 
- **Note:** This folder is cloned directly from the original authors' repository. It contains the original README and codebase.
- For more details, please refer to the original paper: [PTNet3D: A 3D High-Resolution Longitudinal Infant Brain MRI Synthesizer Based on Transformers](https://doi.org/10.1109/TMI.2022.3174827).

### 2. `SFCN/`
Contains the implementation of the **SFCN** (Simple Fully Convolutional Network).
- **Note:** This folder is cloned directly from the original authors' repository. It contains the original README and implementation details.
- Primarily used for downstream clinical evaluation tasks such as **Brain Age Prediction** and **Focal Cortical Dysplasia (FCD) / Epilepsy Classification**.
- For more details, please refer to the original paper: [Accurate brain age prediction with lightweight deep neural networks (Peng et al.)](https://doi.org/10.1016/j.media.2020.101871).

### 3. `notebooks/`
A collection of standalone Jupyter Notebooks used for prototyping, training, and testing various Generative Adversarial Networks and classification models:
- **`eagan.ipynb`**: Implementation of the Edge-aware GAN (Ea-GAN) for medical image translation.
- **`Age Prediction.ipynb`**: Experimental pipeline for brain age regression tasks.
- **`FCD classification.ipynb`**: Pipeline for the binary classification of Focal Cortical Dysplasia.
- **`pix2pix.ipynb`**: Baseline implementation of the Pix2Pix 3D framework for MRI translation tasks.
- **`ptnet.ipynb`**: Experimental notebook for PTNet operations and exploratory analysis.
- **`testing from saved images.ipynb`**: Utility notebook designed to compute comprehensive clinical and perceptual metrics directly from saved NIfTI output volumes.

### 4. `latex thesis/`
Contains the final thesis manuscript and formatting assets.
- **`Thesis.tex`**: The main LaTeX source file of the thesis.
- **`bibliography.bib`**: Reference list and citations.
- **`Images/`**: Contains all figures, network architecture diagrams, and clinical plots used in the document.
- **`Configuration_files/`**: Contains LaTeX formatting, title page, and styling configurations.

## Setup and Requirements
The deep learning models and scripts are implemented in **PyTorch** and utilize **TorchIO** for efficient medical image handling and augmentation. 
Please ensure all dependencies are installed before running the notebooks or training scripts.

**Hardware Note:** The Jupyter notebooks in the `notebooks/` folder are specifically configured to run on Kaggle environments, leveraging the available **NVIDIA Tesla T4 GPU equipped with 16 GB of dedicated VRAM**. Ensure your environment matches or exceeds these specifications for optimal performance.
