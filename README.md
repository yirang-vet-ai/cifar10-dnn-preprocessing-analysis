# Deep Learning DNN Preprocessing Analysis for CIFAR-10

## 1. Overview

This project analyzes how different data preprocessing strategies affect the performance of a Deep Neural Network (DNN) for image classification.

Using the CIFAR-10 dataset, multiple preprocessing pipelines are applied while keeping the same DNN architecture. The goal is to quantitatively evaluate how preprocessing impacts training stability, convergence speed, and generalization performance.

---

## 2. Key Features

- Multiple preprocessing strategies comparison  
- Same DNN model across all experiments (controlled setup)  
- Automatic dataset generation (train / val / test split)  
- GPU-based training (CUDA supported)  
- Performance evaluation and visualization  

---

## 3. Tech Stack

- Python  
- PyTorch  
- OpenCV  
- NumPy  
- Matplotlib  
- CUDA (GPU acceleration)  

---

## 4. Dataset

CIFAR-10 (Canadian Institute for Advanced Research)

- 60,000 images  
- 10 classes  
- 32x32 RGB images  

Classes:
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck

---

## 5. Project Structure

project_root/
├─ cifar10_preprocess_compare_dnn.py
├─ data/
│  └─ cifar10_source_all/
│     ├─ airplane/
│     ├─ automobile/
│     ├─ ...
├─ candidate_data/        # auto-generated
├─ results/               # auto-generated
├─ README.md
├─ LICENSE
├─ NOTICE
├─ requirements.txt
└─ environment.yml

---

## 6. Preprocessing Strategies

- original (baseline)
- CLAHE (contrast enhancement)
- bilateral + CLAHE
- grayscale (3-channel)
- edge detection
- blur + edge detection

---

## 7. Model Architecture

Fully Connected Deep Neural Network (DNN)

- Input: 32×32×3 → flatten (3072)
- Linear → BatchNorm → ReLU → Dropout
- Output: 10 classes

---

## 8. Training Configuration

- Loss: CrossEntropyLoss  
- Optimizer: AdamW  
- Learning Rate: 0.001  
- Batch Size: 64  
- Early Stopping applied  
- LR Scheduler  

---

## 9. How to Run

pip install -r requirements.txt

or

conda env create -f environment.yml
conda activate dl_env

Prepare dataset in:
data/cifar10_source_all/

Run:
python cifar10_preprocess_compare_dnn.py

---

## 10. Output

- candidate_data/
- results/*.pth
- results/*.png
- results/candidate_comparison.txt

---

## 11. Key Insight

Preprocessing is a critical factor influencing model performance, especially in DNN-based image classification.

---

## 12. License

Apache License 2.0

---

## 13. Author

Yirang Jung
