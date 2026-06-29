# 🏦 Loan Default Prediction System

A Machine Learning based web application that predicts whether a loan applicant is likely to default on a loan using Logistic Regression.

---

## 📌 Project Overview

This project helps assess credit risk by predicting loan default based on applicant information such as age, income, loan amount, credit score, employment details, and other financial factors.

The application is built with **Flask** for the backend and uses a trained **Logistic Regression** model from Scikit-learn.

---

## ✨ Features

- Predicts Loan Default (YES/NO)
- Displays probability of default
- Clean and responsive web interface
- Machine Learning model using Logistic Regression
- Real-time prediction through Flask

---

## 🛠 Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML5
- CSS3

---

## 📁 Project Structure

```
Loan-Default-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── Loan_default.csv
│
├── model/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── *.pkl encoders
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
└── notebook/
    └── LOGISTIC REGRESSION.ipynb
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/atharva00infinity-max/Loan-Default-Prediction.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Flask application

```bash
python app.py
```

### 4. Open your browser

```
http://127.0.0.1:5000/
```

---

## 📊 Machine Learning Model

- Algorithm: Logistic Regression
- Feature Scaling: StandardScaler
- Label Encoding for categorical variables
- Probability prediction using `predict_proba()`

---

## 🎯 Future Improvements

- Deploy the application online
- Add more ML models for comparison
- Improve UI/UX
- Add data visualization dashboard

---

## 👨‍💻 Author

**Atharva**

GitHub: https://github.com/atharva00infinity-max