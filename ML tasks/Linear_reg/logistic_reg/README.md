# Logistic Regression (NumPy Implementation with L2 Regularization)

## Overview
This project implements **Logistic Regression** from scratch using **NumPy**, trained with **Gradient Descent** and **L2 Regularization** on a synthetic binary classification dataset.

It evaluates model performance using **Accuracy** and the **ROC Curve**.

---
##  Folder Structure
```
logistic_regression/
│
├── logreg_numpy.py # Main Python script
├── roc_curve.png # ROC curve plot (auto-generated)
├── model_params.npz # Saved model parameters (weights and bias)
├── README.md # Project documentation
└── requirements.txt # Dependencies list
```
---

##  Steps Performed
1. Generate a synthetic binary dataset using `sklearn.datasets.make_classification`
2. Split data into train/test sets
3. Implement sigmoid activation and logistic regression functions manually
4. Train model using **Gradient Descent**
5. Add **L2 regularization** to avoid overfitting
6. Evaluate model with accuracy and ROC-AUC score
7. Plot ROC and Loss curves
8. Save final model parameters

---

## How to Run

1. **Open the folder** in VS Code  
2. Open the integrated terminal and run:

```bash
pip install -r requirements.txt
python logreg_numpy.py
```
## Output files generated
```
roc_curve.png
model_params.npz
```
