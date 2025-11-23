# Fraud Detection using AutoEncoder with PyOD

**Name**  
**Institution Affiliation**  
**GitHub:** [Hands-On Assignment 4](https://github.com/skhanal34053/Hands-On-Assignment-4-Use-Unsupervised-Deep-Learning-Algorithm-to-Detect-Fraud-with-PyOD)  

## Overview
This experiment implements an unsupervised deep learning approach to detect fraudulent credit card transactions using the **AutoEncoder** model from PyOD (Onyeama, 2024). The Kaggle dataset contains 492 frauds out of 284,807 transactions, making it highly imbalanced.

We first checked for missing values; 7 NaNs were found and dropped. Features were standardized, and the AutoEncoder was trained only on normal transactions to learn typical patterns. The model calculates **reconstruction errors**, which are used as **anomaly scores**. Transactions that deviate significantly from normal patterns are flagged as potential fraud.

## Results
- ROC-AUC: 0.955  
- AUPRC: 0.299  
- Classification report shows high precision and recall for normal transactions, and moderate detection for frauds, as expected given the imbalance.  
- The histogram shows normal transactions clustering at low scores, while fraudulent transactions have higher scores, indicating the model captures deviations effectively.

## Conclusion
The PyOD AutoEncoder can detect fraudulent transactions using unsupervised learning. It learns normal patterns and identifies anomalies, making it useful for financial institutions to detect new and unseen fraud patterns (Renström & Holmsten, 2018).

## References
- Onyeama, J. (2024). *Credit Card Fraud Detection in the Nigerian Financial Sector: A Comparison of Unsupervised TensorFlow-Based Anomaly Detection Techniques, Autoencoders and PCA Algorithm*. arXiv preprint arXiv:2407.08758.  
- Renström, M., & Holmsten, T. (2018). *Fraud detection on unlabeled data with unsupervised machine learning.*
