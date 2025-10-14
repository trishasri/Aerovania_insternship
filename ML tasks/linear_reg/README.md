# Linear Regression from Scratch (NumPy)
## Overview
This project implements **linear regression using batch gradient descent** from scratch with **NumPy**.  
It trains on a **synthetic 1-feature dataset**, plots the loss curve, and saves the model parameters.

# Features
- Generates synthetic data with a linear relationship and Gaussian noise.
- Implements **batch gradient descent** to optimize parameters`[bias, weight]`.
- Saves learned parameters to `linreg_params.npy`.
- Plots **loss vs iterations** and saves it as `loss_plot.png`.

## Project Structure
```
linear_regression_submission/
│
├── outputs/                     # Folder to save parameters and plots
│ ├── linreg_params.npy          # Will be created when you run the notebook
│ └── loss_plot.png              # Will be created when you run the notebook
├── linreg_from_scratch.py       # Main notebook (NumPy batch gradient descent)
├── README.md                    # Submission instructions
└── requirements.txt             # Project dependencies
```
## How to run
```
Run the notebook:
Open linreg_from_scratch.ipynb in Jupyter or VS Code notebook interface.
Run all cells sequentially.
```
## Outputs Generated

- **Model parameters**: `outputs/linreg_params.npy`  
- **Loss plot**: `outputs/loss_plot.png`  
- **Console output**: prints the final parameters (`Bias` and `Weight`)

## Example Console Output
```
Training Complete
Final parameters: Bias=-1.218, Weight=4.982
```
The **loss plot** visualizes the MSE decreasing over iterations.
```
The saved **model parameters** can be loaded later to make predictions:

```python
import numpy as np

theta = np.load("outputs/linreg_params.npy")
# theta[0] -> bias, theta[1] -> weight
```