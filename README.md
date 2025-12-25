Vehicles & Animals Image Classification with CIFAR-10

This project implements a complete Convolutional Neural Network (CNN) image classification pipeline using PyTorch, trained on the CIFAR-10 dataset and evaluated on both the standard test set and real-world smartphone images.
The goal is to analyze model performance, generalization ability, and limitations when applied to images outside the training distribution.

Dataset
Standard Dataset

CIFAR-10 (10 classes)

Classes Used:

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

The dataset is automatically downloaded using torchvision.datasets.

Custom Dataset

Real-world images captured using a smartphone and some collected from the internet.
These images are stored in:

dataset/phone image/


All custom images are resized and normalized to match the CIFAR-10 input format.

Project Structure
210112-CN_CIFAR10/
│
├── dataset/
│   └── phone image/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── model/
│   └── 210112.pth
│
├── 210112-CN_CIFAR10.ipynb
├── README.md

Model Architecture

The model is a Convolutional Neural Network implemented using torch.nn.Module.

Key Components

Convolution layers with ReLU activation

MaxPooling layers

Fully connected layers

Output layer with 10 neurons (one per CIFAR-10 class)

Loss Function: CrossEntropyLoss
Optimizer: Adam

Training

Dataset loaded automatically using torchvision.datasets

Images preprocessed using torchvision.transforms

Model trained on the CIFAR-10 training set

Training and validation accuracy and loss recorded across epochs

Training Configuration

Epochs: 10

Batch Size: 64

Device: CPU / GPU (automatically detected)

Training Results

The model was trained for 10 epochs on the CIFAR-10 training set.

Final Epoch Example:

Epoch [10/10] 
Training Loss: ~0.25  
Training Accuracy: ~91%

Observations

Training loss decreases steadily across epochs

Training accuracy increases consistently

Final training accuracy is approximately 91%

Training Plots

The following plots are generated automatically:

Training Loss vs Epochs

Validation Loss vs Epochs

Training Accuracy vs Epochs

Validation Accuracy vs Epochs

These plots demonstrate stable convergence and effective optimization.

Evaluation & Results
Confusion Matrix

A confusion matrix is generated on the CIFAR-10 test dataset to analyze class-wise performance.

Key Observations

Strong performance on structured object classes such as automobile, truck, and ship

Some confusion among visually similar animal classes such as cat, dog, deer, and bird

This behavior is expected due to CIFAR-10’s low resolution (32×32 pixels)

Visual Error Analysis

Three misclassified samples from the CIFAR-10 test set are visualized, showing:

The true label

The predicted label

This analysis helps identify common failure cases and understand model limitations.

Real-World Smartphone Image Predictions

The trained model is evaluated on custom smartphone images stored in dataset/phone image/.

For each image, the system automatically outputs:

Predicted class

Confidence score (Softmax probability)

Observations

Vehicle classes (automobile, truck, ship) are often predicted with high confidence

Animal classes sometimes show confusion due to background, lighting, and pose variations

Confidence scores vary due to domain shift between CIFAR-10 and real-world images

This highlights the generalization limitations of models trained on low-resolution benchmark datasets.

Key Takeaways

The CNN successfully learns visual patterns from CIFAR-10

Training behavior is stable and well-converged

Real-world testing reveals domain differences and generalization challenges

The project demonstrates a complete end-to-end deep learning workflow using PyTorch

How to Run (Google Colab)

Clone the repository:

git clone https://github.com/mitra369/210112-CN_CIFAR10.git


Open 210112-CN_CIFAR10.ipynb in Google Colab.

Select:

Runtime → Run all


No manual file uploads are required.

Author

Antu Mitra
Department of Computer Science and Engineering
