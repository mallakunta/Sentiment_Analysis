# Sentimentanalysis

## Project Overview

This project aims to predict facial expressions using sentiment analysis techniques. The dataset consists of images categorized into seven classes: Ahegao, Angry, Happy, Neutral, Sad, Surprise, and preprocessed. The project leverages deep learning models, specifically Convolutional Neural Networks (CNNs), to classify these expressions accurately.

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [References](#references)

## Dataset

The dataset comprises images divided into the following seven classes:
- Ahegao
- Angry
- Happy
- Neutral
- Sad
- Surprise
- Preprocessed

### Dataset Structure

- Training data shape: (12362, 128, 128, 3)
- Testing data shape: (3091, 128, 128, 3)

## Project Structure

The project is organized as follows:

sentiment-analysis-face-expression
data/
train/
test/
models/
cnn_model.h5
notebooks/
data_preprocessing.ipynb
model_training.ipynb
evaluation.ipynb
src/
preprocess.py
train.py
evaluate.py
README.md
requirements.txt

## Requirements

The project requires the following libraries and tools:

- Python 3.8+
- TensorFlow 2.5+
- NumPy
- Pandas
- scikit-learn
- matplotlib
- OpenCV

Install the required packages using:

```bash
pip install -r requirements.txt


