# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:12:10 2025

@author: LAB
"""

import streamlit as st
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_blobs


st.set_page_config(page_title="K-Means Clustering", layout="centered")

# Load KMeans model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Set the title
st.title("K-Means Clustering Visualizer by Saw San Nyunt Win")

# Generate synthetic dataset
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict using the loaded model
y_kmeans = loaded_model.predict(X)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], 
           s=300, c='red')  # no label
ax.set_title('K-Means Clustering')
# ax.legend()  ← do NOT call this
st.pyplot(fig)

