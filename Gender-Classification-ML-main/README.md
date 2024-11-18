## Project Summary

This project explores the Gender Classification Dataset from Kaggle, applying various machine learning models to classify gender based on image data. The process includes data preparation, multiple experiments with machine learning algorithms, and a final evaluation to identify the best-performing model.

## Key Steps
### 1. Data Preparation

* All images were reshaped to a uniform dimension of (64, 64, 3).
* The RGB images were converted to grayscale and normalized for further processing.

### 2. Experiments and Models

* **First Experiment:** A Support Vector Machine (SVM) model was trained on grayscale images, and performance was evaluated using confusion matrix and F1 scores.
* **Second Experiment:** Two different neural network architectures were developed and tested on grayscale images, with model performance tracked through error and accuracy curves. The best-performing model was saved, reloaded, and tested.
* **Third Experiment:** A Convolutional Neural Network (CNN) was trained on both grayscale and RGB images, with model performance similarly tracked. The best CNN model was saved, reloaded, and tested.

### 3. Model Evaluation

* All models were compared based on their F1 scores, accuracy, and confusion matrices.

## Dataset

The dataset used for this project can be found on Kaggle:
[Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset)
