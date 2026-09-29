# Master's Thesis Repository

This repository contains the complete source code, experimental notebooks, and final documentation for my Master's Thesis. 

## Repository Structure

### 1. `PTNet3D/`
Contains the source code for the **PTNet3D** architecture (3D High-Resolution Longitudinal Infant Brain MRI Synthesizer Based on Transformers). 
- Includes custom data loaders, network definitions, training routines, and evaluation metrics.

### 2. `SFCN/`
Contains the implementation of the **SFCN** (Simple Fully Convolutional Network).
- Primarily used for downstream clinical evaluation tasks such as **Brain Age Prediction** and **Focal Cortical Dysplasia (FCD) / Epilepsy Classification**.
- Includes cross-validation scripts and model weights evaluated on multiple datasets.

### 3. `Downloads_Codes/`
A collection of standalone Jupyter Notebooks used for prototyping, training, and testing various Generative Adversarial Networks and classification models:
- **`eagan.ipynb`**: Implementation of the Edge-aware GAN (Ea-GAN) for medical image translation.
- **`Age Prediction.ipynb`**: Experimental pipeline for brain age regression tasks.
- **`FCD classification.ipynb`**: Pipeline for the binary classification of Focal Cortical Dysplasia.
- **`pix2pix.ipynb`**: Baseline implementation of the Pix2Pix 3D framework for MRI translation tasks.
- **`ptnet.ipynb`**: Experimental notebook for PTNet operations and exploratory analysis.
- **`testing from saved images.ipynb`**: Utility notebook designed to compute comprehensive clinical and perceptual metrics directly from saved NIfTI output volumes.

### 4. `TESI_PRONTA_DA_CARICARE/`
Contains the final thesis manuscript and formatting assets.
- **`Thesis.tex`**: The main LaTeX source file of the thesis.
- **`bibliography.bib`**: Reference list and citations.
- **`Images/`**: Contains all figures, network architecture diagrams, and clinical plots used in the document.
- **`Configuration_files/`**: Contains LaTeX formatting, title page, and styling configurations.

## Setup and Requirements
The deep learning models and scripts are implemented in **PyTorch** and utilize **TorchIO** for efficient medical image handling and augmentation. 
Please ensure all dependencies are installed before running the notebooks or training scripts.
