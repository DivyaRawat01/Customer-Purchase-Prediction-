# 🛒 Customer Purchase Prediction System

This project predicts whether a customer is likely to make a purchase based on demographic and behavioral data using machine learning.
The final model is deployed with an interactive Gradio web interface and can be shared publicly.

## 📌 Project Overview

**Problem Type:** Binary Classification

**Target Variable:** PurchaseStatus (0 = No, 1 = Yes)

**Best Model Used:** Random Forest Classifier

**Deployment:** Gradio (Hugging Face compatible)

## 📊 Dataset Description

The dataset contains the following features:

**Column	Description**
Age	Customer’s age
Gender	0 = Male, 1 = Female
AnnualIncome	Annual income in dollars
NumberOfPurchases	Total purchases made
ProductCategory	0: Electronics, 1: Clothing, 2: Home Goods, 3: Beauty, 4: Sports
TimeSpentOnWebsite	Time spent on website (minutes)
LoyaltyProgram	0 = No, 1 = Yes
DiscountsAvailed	Number of discounts used (0–5)
PurchaseStatus	Target variable (0 = No, 1 = Yes)

A realistic synthetic dataset was generated using logical business rules to simulate real-world customer behavior.

## 🔍 Exploratory Data Analysis (EDA)

Checked data types and missing values

Analyzed target distribution

Visualized relationships between features and purchase behavior

## 🔄 Data Preprocessing

Label encoding for categorical features

Feature scaling using StandardScaler

Train–test split

## 🤖 Models Implemented

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Random Forest **(Best Performance)**

## 📈 Model Evaluation

Models were evaluated using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix

Random Forest achieved the best overall performance with realistic accuracy.

## 🚀 Deployment

The trained model was deployed using Gradio, allowing users to input customer details and get real-time predictions.

Features of the App:

User-friendly input form

Predict button for inference

Clear output message (Purchase Likely / Unlikely)

Public shareable link

--

## 📂 Project Structure

`
project/

│── app.py

│── purchase_prediction_model.pkl

│── scaler.pkl

│── feature_columns.pkl

│── requirements.txt

│── README.md
`

## ⚙️ Installation & Usage
Install dependencies

`
pip install -r requirements.txt
`

- Run the app

`
python app.py
`

To generate a public link:

interface.launch(share=True)

## 🧪 Technologies Used
---
- Python
- Pandas
- NumPy
- Scikit-learn
- Gradio
- Joblib

--

## 📝 Conclusion

This project demonstrates an end-to-end machine learning workflow — from data preprocessing and model training to evaluation and deployment.
It highlights the importance of data quality, proper feature encoding, and model selection in classification tasks.
---

## 👤 Author
Divya
