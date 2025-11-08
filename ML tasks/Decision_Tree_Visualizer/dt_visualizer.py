import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

#Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\winemag-data-130k-v2.csv")

# Keep a few useful columns (numeric + categorical)
# Keep selected columns (remove text-heavy ones)
cols = ["points", "price", "country", "province", "variety"]
df = df[cols].dropna()

# Keep only top 5 frequent target classes to avoid stratify error
top_classes = df["country"].value_counts().nlargest(5).index
df = df[df["country"].isin(top_classes)]

# Encode categorical columns
for col in ["country", "province", "variety"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define features and target
# We'll predict the "country" (multi-class) based on other features
X = df.drop(columns=["country"])
y = df["country"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

# Save model and tree image
joblib.dump(clf, "decision_tree_model.joblib")
joblib.dump(scaler, "outputs/scaler.joblib")

# Plot decision tree
plt.figure(figsize=(25, 12))
plot_tree(
    clf,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=200)
plt.close()
print("Saved decision_tree.png and model")
