# PCA
# CIFAR-10 Image Classification with PCA and Neural Network
# by Sara Afshar

This project demonstrates image classification on the CIFAR-10 dataset using Principal Component Analysis (PCA) and a Neural Network model. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes. The project involves loading and preprocessing the dataset, applying PCA for dimensionality reduction, and training a neural network model to classify the images. The model's performance is evaluated using metrics such as accuracy and confusion matrix, and the results are visualized to provide insights into the classification process.

## Overview

The project performs the following steps:
1. Load and preprocess the CIFAR-10 dataset.
2. Implement PCA for dimensionality reduction.
3. Train a Neural Network model on the PCA-transformed data.
4. Evaluate the model's performance using metrics such as accuracy and confusion matrix.
5. Visualize the model's training history and confusion matrix.

## Installation

To run this project, you'll need to have Python installed along with the following libraries:

- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn
- tensorflow

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn seaborn tensorflow
