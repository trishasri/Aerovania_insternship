 ## K-Means Clustering from Scratch (with K-Means++ Initialization)
 This project implements the **K-Means clustering algorithm from scratch using NumPy**, including **optional K-Means++ initialization**.  
The task is performed on **2D synthetic data**, and results are evaluated using **Inertia** and **Silhouette Score (via scikit-learn)**.  
A cluster plot with centroids is generated and exported as a PNG file.
---
## Features Implemented
From-scratch K-Means clustering  
K-Means++ centroid initialization  
Works on 2D dataset (generated using `make_blobs`)  
Inertia calculation  
Silhouette Score (using `sklearn.metrics.silhouette_score`)  
 Final cluster visualization with centroids  
PNG export of cluster plot  
Centroids saved as `.npz` file 
---
##  Project Structure
```
Kmeans
├── kmeans_scratch.ipynb
├── requirements.txt
├── README.md
└── (generated after running)
    ├── cluster_plot.png
    └── model_centroids.npz
```
---
## How to Run
```
Install dependencies:
   pip install -r requirements.txt
```
---
## Run all cells to:
   - Train the K-Means model
   - Print inertia & silhouette score
   - Generate cluster_plot.png
   - Save centroids in model_centroids.npz
---
## Output Files
- cluster_plot.png
- model_centroids.npz