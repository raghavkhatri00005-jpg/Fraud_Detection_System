# 💳 Financial Fraud Detection Using Machine Learning

A simple Machine Learning project that detects potentially fraudulent financial transactions using transaction-related features such as transaction amount, transaction frequency, distance, and night-time activity.

## 📌 Problem Statement

Financial fraud detection is an important task for financial institutions. The goal of this project is to classify transactions as:

- `0` → Genuine transaction
- `1` → Fraudulent transaction

Since fraud datasets are usually highly imbalanced, this project focuses on **Precision, Recall, and F1-Score** rather than accuracy alone.

## 🎯 Objectives

- Load and preprocess transaction data
- Handle missing values and duplicate records
- Handle imbalanced fraud data
- Perform Exploratory Data Analysis (EDA)
- Train Machine Learning models
- Compare model performance
- Reduce false positives using a probability threshold
- Visualize the results
- Identify important features for fraud detection

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 📂 Project Structure

```text
Financial-Fraud-Detection/
│
├── fraud_detection_simple.py
├── creditcard.csv              # Optional real dataset
├── README.md
│
└── output/
    ├── class_distribution.png
    ├── correlation.png
    ├── confusion_matrix.png
    └── feature_importance.png
