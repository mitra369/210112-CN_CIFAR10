# 210112-CNN
CNN Image Classification using CIFAR-10 (PyTorch)

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images into four categories: Bird, Car, Cat, and Dog.
The model is trained on a subset of the CIFAR-10 dataset and evaluated using real-world images captured with a smartphone.

Project Objectives

Build a complete CNN-based image classification pipeline using PyTorch

Train the model on a standard dataset (CIFAR-10)

Test the trained model on custom phone images

Ensure full automation using GitHub and Google Colab

Visualize model performance using plots and confusion matrix

Dataset
Standard Dataset

CIFAR-10 dataset (loaded via torchvision.datasets)

Selected classes:

Bird

Car

Cat

Dog

Custom Dataset

Real-world images captured using a smartphone

Stored inside the following directory:

dataset/phone image/


Images are resized and normalized to match the CIFAR-10 data format

Model Architecture

The CNN model consists of:

Two convolutional layers

ReLU activation functions

Max pooling layers

Fully connected layers

Softmax output for probability prediction

The model is implemented using the PyTorch framework.

Training Details

Loss Function: CrossEntropyLoss

Optimizer: Adam

Epochs: 10

Batch Size: 64

Device: CPU or GPU (automatically detected)

Results and Visualizations

The notebook generates the following outputs:

Training loss vs epoch plot

Validation accuracy vs epoch plot

Confusion matrix on the test dataset

Visual error analysis showing misclassified images

Prediction gallery of phone images with confidence scores

Example prediction format:

Predicted Class: Cat (97.8%)

Repository Structure
210112-CN_CIFAR10/
│
├── dataset/
│   └── phone image/
│       ├── car1.jpg
│       ├── cat1.jpg
│       ├── dog1.jpg
│       └── ...
│
├── model/
│   └── 210112.pth
│
├── 210112-CN_CIFAR10.ipynb
├── README.md

How to Run (Google Colab)

Open the notebook in Google Colab

Select Runtime → Run all

The notebook will automatically:

Clone the GitHub repository

Download the CIFAR-10 dataset

Train or load the CNN model

Process custom phone images

Display all required visual outputs

No manual file uploads are required.

Key Features

Fully automated workflow

Real-world image testing

Standard evaluation metrics and visualizations

Assignment-compliant implementation

Author

Antu Mitra
Department of Computer Science and Engineering
Jashore University of Science and Technology
