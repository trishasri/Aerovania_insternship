import numpy as np                      # For numerical operations (vectors, matrices, math)
import matplotlib.pyplot as plt         # For plotting graphs like ROC and loss curves
from sklearn.datasets import make_classification  # For creating synthetic binary classification datasets
from sklearn.model_selection import train_test_split  # To split dataset into train and test sets
from sklearn.metrics import accuracy_score, roc_curve, auc # For evaluating accuracy and ROC/AUC

#Generate synthetic binary classification dataset
X, y = make_classification(
    n_samples=500,       # 500 samples (rows)
    n_features=2,        # 2 input features
    n_informative=2,     # both features are useful
    n_redundant=0,       # no redundant features
    n_clusters_per_class=1,
    class_sep=1.5,       # moderate separation between classes
    random_state=42      # for reproducibility
)

# Split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Helper functions for logistic regression
# Sigmoid function
def sigmoid(z):
    """Sigmoid activation function that maps any real value to (0,1)."""
    return 1 / (1 + np.exp(-z))

# #Loss function with L2 regularization
def compute_loss(X, y, weights, bias, lambda_reg):
    """
    Compute binary cross-entropy loss with L2 regularization.
    """
    n = X.shape[0]                                # number of samples
    linear_model = np.dot(X, weights) + bias       # linear combination
    y_pred = sigmoid(linear_model)                 # predicted probabilities
    
    # Avoid log(0) using small epsilon
    eps = 1e-9
    # Binary cross-entropy loss
    loss = - (1 / n) * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
    
    # Add L2 regularization penalty
    reg_term = (lambda_reg / (2 * n)) * np.sum(weights ** 2)
    return loss + reg_term

# Gradient function
def compute_gradients(X, y, weights, bias, lambda_reg):
    """
    Compute gradients for weights and bias.
    """
    n = X.shape[0]
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    
    # Gradient of weights includes regularization term
    dw = (1 / n) * np.dot(X.T, (y_pred - y)) + (lambda_reg / n) * weights
    db = (1 / n) * np.sum(y_pred - y)
    return dw, db

# Initialize model parameters
weights = np.zeros(X_train.shape[1])   # initialize weights (vector of zeros)
bias = 0                               # initialize bias term
learning_rate = 0.1                    # step size for gradient descent
lambda_reg = 0.1                       # regularization strength
epochs = 1000                          # number of iterations
losses = []                            # to store loss values for plotting

# Training loop using Gradient Descent
for epoch in range(epochs):
    # Compute gradients for current weights and bias
    dw, db = compute_gradients(X_train, y_train, weights, bias, lambda_reg)
    
    # Update parameters (gradient descent step)
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    # Compute and store loss for visualization
    loss = compute_loss(X_train, y_train, weights, bias, lambda_reg)
    losses.append(loss)
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# Evaluate model on test data
# Predict probabilities for test set
y_prob = sigmoid(np.dot(X_test, weights) + bias)

# Convert probabilities to class labels (threshold 0.5)
y_pred = (y_prob >= 0.5).astype(int)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {accuracy * 100:.2f}%")

# Plot ROC curve and save as PNG
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression (NumPy)')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve.png', dpi=300)   # save image to project folder
plt.show()

# Save trained model parameters
np.savez('model_params.npz', weights=weights, bias=bias)
print("\n Model parameters saved as model_params.npz")
