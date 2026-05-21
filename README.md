# Brain Tumor Segmentation & 3D Visualization

## Demo

![Demo](assets/[Segmentation/SEG/Screenshot (971)](https://github.com/maskar122/Brain-Tumor-Segmentation-3D-Visualization/blob/136d86601f2525c36e6de7974cbe59329f1676c7/Segmentation/SEG/Screenshot%20(971).png)

A Deep Learning-based medical imaging project for automated brain tumor segmentation and interactive 2D/3D visualization using MRI scans from the BraTS2020 dataset.

The system utilizes **DynUNet 3D** architecture with the **MONAI framework** and **PyTorch** to perform accurate volumetric tumor segmentation and generate interactive visualizations for medical analysis.

---

# Features

- 3D Brain Tumor Segmentation
- MRI Volumetric Processing
- Interactive 2D Visualization
- Interactive 3D Mesh Visualization
- Automated Tumor Detection
- Deep Learning-based Inference
- GPU Acceleration Support

---

# Technologies Used

- Python
- PyTorch
- MONAI
- DynUNet 3D
- NumPy
- NiBabel
- Plotly
- Matplotlib

---

# Dataset

This project uses the **BraTS2020 (Brain Tumor Segmentation Challenge 2020)** dataset.

The dataset contains multimodal MRI scans including:

- T1
- T1Gd
- T2
- FLAIR

along with ground truth segmentation masks for brain tumors.

---

# Model Architecture

The segmentation model is based on:

- DynUNet 3D
- Encoder-Decoder Architecture
- Volumetric Convolutional Layers
- MONAI Medical Imaging Pipeline

---

# Visualization

## 2D Visualization

- MRI Slice Inspection
- Predicted Mask Overlay
- Ground Truth Comparison

## 3D Visualization

- Tumor Mesh Reconstruction
- Interactive Volumetric Rendering
- 3D Medical Visualization using Plotly

---

# Project Structure

```bash
├── assets/
├── data/
├── models/
├── notebooks/
├── training/
├── inference/
├── visualization/
├── outputs/
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Training

```bash
python train.py
```

---

# Run Inference

```bash
python inference.py
```

---

# Results

The model performs automated segmentation of brain tumor regions from MRI volumes and enables interactive 2D and 3D visualization for enhanced medical image analysis.

---

# Future Improvements

- Transformer-based Segmentation Models
- Real-time Visualization
- Web-based Deployment
- Multi-class Tumor Segmentation
- Clinical Reporting Integration

---

# Author

Mohamed Essam Askar

AI Engineer | Computer Vision & Medical Imaging

## Links

- LinkedIn: https://www.linkedin.com/
- GitHub: https://github.com/
