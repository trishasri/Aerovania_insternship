import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Synthetic Data Loading
np.random.seed(42)
X = 2 * np.random.rand(200, 1) # 200 samples, 1 feature
true_w, true_b = 5, -1.2
y = true_w * X[:, 0] + true_b + np.random.randn(200) * 0.5  # Add noise

# Data Preprocessing
X_b = np.c_[np.ones((len(X), 1)), X] # shape -> (200, 2)

# Model Initialization
theta = np.random.randn(2) # [bias, weight]
learning_rate = 0.1
n_iterations = 200
losses = []

# Batch Gradient Descent
for i in range(n_iterations):
    # Predictions
    y_pred = X_b.dot(theta)
    
    # Compute error
    error = y_pred - y
    
    # Compute MSE loss
    loss = (1 / (2 * len(X))) * np.sum(error**2)
    losses.append(loss)
    
    # Compute gradients
    gradients = (1 / len(X)) * X_b.T.dot(error)
    
    # Update parameters
    theta -= learning_rate * gradients

# Save Model Parameters
np.save("outputs/linreg_params.npy", theta)
print("Training Complete")
print(f"Final parameters: Bias={theta[0]:.3f}, Weight={theta[1]:.3f}")

# Plot Loss Curve
plt.plot(losses)
plt.title("Loss vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("outputs/loss_plot.png")
plt.show()