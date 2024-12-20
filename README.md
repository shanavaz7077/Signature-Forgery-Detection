# Signature Forgery Detection Using CNN

## Overview

This project focuses on detecting forged signatures using advanced preprocessing techniques and Convolutional Neural Networks (CNN). It addresses the increasing need for robust signature verification in various sectors like banking, legal, and documentation. By leveraging a combination of image preprocessing and deep learning, the system ensures high accuracy in identifying forged signatures.

## Features

- **Comprehensive Preprocessing Pipeline**:

  - Noise reduction using advanced denoising algorithms.
  - Grayscale conversion and binarization for simplified analysis.
  - Contour-based region of interest (ROI) extraction for focusing on signature areas.
  - Thinning techniques to enhance the structural integrity of signatures.

- **Deep Learning Model**:

  - A custom CNN model for binary classification of signatures (genuine vs forged).
  - Incorporates data augmentation for improved model generalization.
  - Metrics-driven evaluation using confusion matrix, precision, recall, and F1-score.

- **Adaptability**:
  - Handles diverse datasets with varying signature styles and formats.
  - Preprocessing and augmentation tailored to improve performance across datasets.

## Table of Contents

- [Signature Forgery Detection Using CNN](#signature-forgery-detection-using-cnn)
  - [Overview](#overview)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
  - [Usage](#usage)
    - [Preprocessing](#preprocessing)
    - [Training and Testing](#training-and-testing)
  - [Model Architecture](#model-architecture)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Contributing](#contributing)
  - [License](#license)

---

## Project Structure

```
├── preprocess.py             # Preprocessing module for cleaning and preparing images
├── signature_detection.py    # CNN model for signature classification
├── datasets/                 # Directory for training and testing datasets
│   ├── train/
│   └── test/
├── outputs/                  # Directory for storing processed images and results
└── README.md                 # Documentation for the project
```

---

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- TensorFlow
- OpenCV
- scikit-image
- NumPy
- Matplotlib
- Seaborn

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/signature-forgery-detection.git
   cd signature-forgery-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Preprocessing

1. Place your raw images in the input folder specified in `preprocess.py`.
2. Run the preprocessing script to clean and prepare the images:
   ```bash
   python preprocess.py
   ```
3. Processed images will be saved in the output directory.

### Training and Testing

1. Ensure the datasets are organized in the `datasets/train` and `datasets/test` directories.
2. Train the CNN model using the following command:
   ```bash
   python signature_detection.py
   ```
3. The model will output predictions and evaluation metrics, including a confusion matrix and classification report.

---

## Model Architecture

The CNN model includes:

1. **Convolutional Layers**: Extract features from input images.
2. **Max-Pooling Layers**: Downsample feature maps for computational efficiency.
3. **Fully Connected Layers**: Classify signatures as genuine or forged.
4. **Sigmoid Activation**: Output probabilities for binary classification.

---

## Evaluation Metrics

- **Confusion Matrix**: Visual representation of classification performance.
- **Precision**: Fraction of correctly identified positive cases.
- **Recall**: Fraction of true positive cases among all actual positive cases.
- **F1-Score**: Harmonic mean of precision and recall.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
