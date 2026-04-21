# Single Image Super Resolution using CNN and GAN
---

## Overview
This project implements **Single Image Super Resolution (SISR)**, where a
high‑resolution (HR) image is reconstructed from a single low‑resolution (LR)
input image. The goal is to enhance image quality by recovering fine details
lost during downscaling.

Two deep learning approaches are implemented and compared:
- **SRCNN** – a convolutional neural network (baseline approach)
- **SRGAN** – a generative adversarial network for perceptual super‑resolution

This project is developed as a **college minor project** and focuses on
understanding and comparing CNN‑based and GAN‑based super‑resolution methods.

---

## Project Structure

```
Single-Image-Super-Resolution/
│
├── notebook/
│   └── sisr_project.ipynb
│
├── src/
│   ├── dataset.py
│   ├── srcnn_model.py
│   └── srgan_models.py
│
├── scripts/
│   └── inference.py
│
├── sample_images/
│   └── input_lr.png
│
├── results/
│   └── comparison.png
│
├── README.md
├── requirements.txt
└── .gitignore


```

---


## Models Used

### SRCNN (Super‑Resolution Convolutional Neural Network)
SRCNN is a baseline convolutional neural network that learns a direct mapping
between low‑resolution and high‑resolution images.

**Key characteristics:**
- Three convolutional layers
- Mean Squared Error (MSE) loss
- Simple and lightweight architecture
- Faster training but limited perceptual quality

SRCNN is used as a baseline to compare against GAN‑based super‑resolution.

---

### SRGAN (Super‑Resolution Generative Adversarial Network)
SRGAN improves visual quality using adversarial learning and perceptual loss.

**Components:**
- **Generator**: Converts low‑resolution images into high‑resolution images
- **Discriminator**: Differentiates between real and generated images
- **Perceptual loss**: Computed using features from a pretrained VGG19 network

**Advantages over SRCNN:**
- Sharper textures
- Better perceptual realism
- Improved visual quality

---

## Technology Stack
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Computer Vision**: OpenCV
- **Numerical Computing**: NumPy
- **Visualization**: Matplotlib
- **Development Environment**: VS Code with Jupyter Notebook

---

## Setup and Usage

### 1. Clone the repository
```bash
git clone https://github.com/KDh3h3/Single-Image-Super-Resolution.git
cd Single-Image-Super-Resolution
```

### 2. Create and activate virtual environment 
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```
### 3: Install required dependencies
```bash
pip install -r requirements.txt
```
### 4: Run the project using Jupyter Notebook
```bash
notebook/sisr_project.ipynb
```
---

## Results
- The final output compares:
- Low‑resolution input
- SRCNN output
- SRGAN output
- Ground‑truth high‑resolution image

The comparison image is saved as:
```bash
results/comparison.png
```
## Observation:
- SRCNN produces smoother outputs
- SRGAN generates sharper and more visually realistic images

## Conclusion
This project demonstrates the application of deep learning techniques for
image super‑resolution and highlights the difference between CNN‑based and
GAN‑based approaches. While SRCNN provides a strong baseline, SRGAN achieves
significantly better perceptual quality, making it more suitable for realistic
super‑resolution tasks.