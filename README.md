# capstone_project
# Satellite Image Segmentation using U-Net Architecture

## Project Overview
This repository contains the **Capstone Project** for the Deep Learning course. The project addresses a core problem in computer vision and remote sensing: **semantic image segmentation**. 

Using a custom-built **U-Net architecture** implemented from scratch in **PyTorch**, the model is trained to perform pixel-wise classification on satellite/aerial imagery to detect structural objects (such as buildings and infrastructure) against a complex geographical background.

---

## Tech Stack & Key Concepts
- **Framework:** PyTorch, Torchvision
- **Architecture:** U-Net (Contracting Path, Expansive Path, and Skip Connections)
- **Data Engineering:** Automated synthetic geospatial data generation, pixel-wise mask structuring
- **Metrics:** Cross-Entropy Loss, **Intersection over Union (IoU)**
- **Visualization:** Matplotlib, Seaborn

---

## Model Architecture (U-Net)
The implemented model strictly follows the encoder-decoder structure of the classic U-Net:
1. **Encoder (Contracting Path):** Extracts high-level feature maps using stacked alternative `3x3` convolutions, Batch Normalization, ReLU activations, and `2x2` Max Pooling layers.
2. **Decoder (Expansive Path):** Reconstructs the spatial resolution using `2x2` Transposed Convolutions (`nn.ConvTranspose2d`).
3. **Skip Connections:** Concatenates high-resolution features from the contracting path directly into the decoder to preserve fine structural boundaries and precise pixel localization.

---

## Evaluation & Metrics
Standard classification accuracy is insufficient for semantic segmentation due to extreme class imbalance (background pixels significantly outnumber object pixels). To counter this, the project evaluates performance using the **Intersection over Union (IoU)** metric:

$$\text{IoU} = \frac{\text{Target} \cap \text{Prediction}}{\text{Target} \cup \text{Prediction}}$$

### Key Achievements:
- Successfully implemented precise skip-connection tensor concatenation (`torch.cat`) in PyTorch.
- Tracked Loss and IoU metrics across training/validation epochs to optimize performance and prevent overfitting.
- Visualized edge preservation and semantic boundaries in predicted masks compared to Ground Truth targets.

---

## How to Run the Project
1. Clone this repository to your local machine or open it directly in environment.
2. Open the main `.ipynb` notebook file in **Google Colab** or Jupyter Notebook.
3. Ensure you have the required dependencies installed:
   ```bash
   pip install torch torchvision numpy matplotlib pillow scikit-learn seaborn
