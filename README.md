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

Directory:

---

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


## Evaluation & Results

---<img width="547" height="415" alt="download" src="https://github.com/user-attachments/assets/09227334-58da-43c3-8dd1-a03e4012e958" />
<img width="556" height="413" alt="download" src="https://github.com/user-attachments/assets/c0e3eaab-dae3-45f4-9430-c2bae8178722" />

### Confusion Matrix

A confusion matrix was generated on the CIFAR-10 test set to analyze per-class performance.
<img width="853" height="743" alt="download" src="https://github.com/user-attachments/assets/d635445f-e529-499d-99d1-1d4cceaef4b9" />


### Visual Error Analysis

Three misclassified test images were visualized with true and predicted labels.

---

<img width="717" height="268" alt="download" src="https://github.com/user-attachments/assets/ef21abcc-15ac-4234-9df8-83a09b9b5c1a" />

## Real-World Smartphone Image Predictions

The trained model was evaluated on custom smartphone images.

For each image:
- Predicted class
- Confidence score

---
<img width="1490" height="1490" alt="download" src="https://github.com/user-attachments/assets/d278b56b-27ed-4f42-aa44-b8444409f4c5" />


## How to Run (Google Colab)

```bash
git clone https://github.com/mitra369/210112-CN_CIFAR10.git

---

## VERY IMPORTANT CHECKLIST (THIS IS WHY IT FAILED BEFORE)

Make sure **ALL** are true:

1. File name is **README.md** (not `.txt`)
2. The `#` symbol is in **column 1** (no space before it)
3. You are viewing it on **GitHub repository page**, not Notepad or Word
4. You did **not** paste it inside a code block accidentally

---

If you want, next I can:
- Convert your **existing text** automatically into Markdown
- Check your GitHub repo live and tell you what is wrong
- Make it look exactly like a published research repo

Just tell me.
