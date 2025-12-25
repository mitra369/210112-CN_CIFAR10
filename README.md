# Vehicles & Animals Image Classification with CIFAR-10

This project implements a complete Convolutional Neural Network (CNN) image classification pipeline using PyTorch, trained on the CIFAR-10 dataset and evaluated on both the standard test set and real-world smartphone images.

---

## Dataset

### Standard Dataset

**CIFAR-10 (10 classes)**

### Classes Used
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

### Custom Dataset

Real-world smartphone images captured by the author and collected from the internet.

### Project Structure:

---
<img width="620" height="332" alt="Screenshot 2025-12-17 130301" src="https://github.com/user-attachments/assets/7eb7085d-e161-4235-a459-ed9a2e2e989d" />

## Model Architecture

Convolutional Neural Network implemented using `torch.nn.Module`.

### Key Layers
- Convolution + ReLU
- MaxPooling
- Fully Connected layers

**Loss Function:** CrossEntropyLoss  
**Optimizer:** Adam  

---

## Training

- Dataset downloaded automatically using `torchvision.datasets`
- Images preprocessed using `torchvision.transforms`
- Model trained on CIFAR-10 training set
- Training and validation metrics recorded across epochs


**Epochs:** 10  
**Batch Size:** 64  
## Training Log (Per Epoch)
<img width="495" height="227" alt="Screenshot 2025-12-26 025004" src="https://github.com/user-attachments/assets/d0f33cf3-3daf-4800-836b-5c4cd236ea8e" />


## Validation Results:
### Loss vs Epoch

<img width="547" height="417" alt="download" src="https://github.com/user-attachments/assets/85475b7a-1708-4a31-91c5-d7f1cc300c08" />

### Accuracy vs Epoch

<img width="556" height="413" alt="download" src="https://github.com/user-attachments/assets/edcd6a2a-0efb-453c-b4c2-a2ecf115fd5a" />

### Observations
Training loss decreases steadily
Training accuracy improves consistently
Validation accuracy stabilizes around 70–73%
Overfitting begins after later epochs (expected behavior)

### Confusion Matrix

A confusion matrix was generated on the CIFAR-10 test set to analyze per-class performance.

<img width="576" height="484" alt="download" src="https://github.com/user-attachments/assets/23821f50-ff87-452f-a93b-3e20eaa08d0f" />


### Visual Error Analysis

Three misclassified test images were visualized with true and predicted labels.

---
<img width="717" height="268" alt="download" src="https://github.com/user-attachments/assets/7c213e00-dc4a-47fb-ac15-bf6db6c850c3" />



### Real-World Smartphone Image Predictions

The trained model was evaluated on custom smartphone images.

For each image:
- Predicted class
- Confidence score

---

<img width="1490" height="1490" alt="download" src="https://github.com/user-attachments/assets/6882d281-f8aa-45b6-a146-7bbcdbd14f4c" />

### Observations
Vehicle images are classified with high confidence
Animal classes occasionally show confusion
Confidence varies due to domain shift between CIFAR-10 and real-world images

### How to Run (Google Colab)

Open the notebook in Google Colab:
Click here to open 210112_CN_CIFAR10.ipynb in Colab

Select Runtime → Run all

All training, evaluation, and visualizations will be executed automatically.
No manual cloning or file uploads are required.

## Author
### Antu Mitra
### Student ID: 210112
### Department of Computer Science and Engineering
### Jashore University of Science and Technology
