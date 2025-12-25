Vehicles & Animals Image Classification with CIFAR-10

This project implements a complete Convolutional Neural Network (CNN) image classification pipeline using PyTorch, trained on the CIFAR-10 dataset and evaluated on both the standard test set and real-world smartphone images.

Dataset
Standard Dataset

CIFAR-10 (10 classes)

Classes Used

airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck

Custom Dataset

Real-world smartphone images captured by the author and collected from the internet.

Directory:

dataset/phone image/

Project Structure
210112-CN_CIFAR10/
├── dataset/
│   └── phone image/
├── model/
│   └── 210112.pth
├── 210112-CN_CIFAR10.ipynb
└── README.md

Model Architecture

Convolutional Neural Network implemented using torch.nn.Module.

Key Layers

Convolution + ReLU

MaxPooling

Fully Connected layers

Loss Function: CrossEntropyLoss
Optimizer: Adam

Training

Dataset downloaded automatically using torchvision

Images preprocessed using torchvision.transforms

Model trained on CIFAR-10 training set

Training and validation metrics recorded across epochs

Epochs: 10
Batch Size: 64

Training Results

Final epoch results:

Epoch [10/10]
Training Loss ≈ 0.25
Training Accuracy ≈ 91%

Evaluation & Results
Confusion Matrix

A confusion matrix was generated on the CIFAR-10 test set to analyze per-class performance.

Visual Error Analysis

Three misclassified test images were visualized with true and predicted labels.

Real-World Smartphone Image Predictions

The trained model was evaluated on custom smartphone images.

For each image:

Predicted class

Confidence score

Key Takeaways

CNN learns CIFAR-10 visual patterns effectively

Stable convergence during training

Domain shift affects real-world image performance

How to Run (Google Colab)
git clone https://github.com/mitra369/210112-CN_CIFAR10.git


Open the notebook and select:

Runtime → Run all

Author

Antu Mitra
Department of Computer Science and Engineering
