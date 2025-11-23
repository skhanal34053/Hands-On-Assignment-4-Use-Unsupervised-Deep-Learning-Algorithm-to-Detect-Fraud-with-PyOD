# Fraud Detection using AutoEncoder with PyOD

## Overview
This project uses an **unsupervised AutoEncoder model** from PyOD to detect fraudulent credit card transactions in a highly imbalanced dataset from Kaggle.

## Results
- **ROC-AUC:** 0.955  
- **AUPRC:** 0.299  
- High precision and recall for normal transactions; moderate detection for frauds.  
- Histogram shows normal transactions cluster at low scores, fraudulent ones at higher scores.

## Key Insights
- Reconstruction errors effectively identify anomalies.  
- Model learns normal patterns and flags deviations as potential frauds.  
- Unsupervised learning is suitable for detecting new, unseen fraud patterns.

## Repository Contents
- `creditcard.csv` – Kaggle dataset 
- `Autoencoder fraud detection.py` – Complete Python script for training and evaluation  
- `README.md` 
