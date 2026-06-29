from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load Model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load Encoders
edu_encoder = joblib.load("Education_encoder.pkl")
emp_encoder = joblib.load("employment_encoder.pkl")
mar_encoder = joblib.load("maritalStatus_encoder.pkl")
mort_encoder = joblib.load("hasmortgage_encoder.pkl")
purp_encoder = joblib.load("loanpurpose_encoder.pkl")
cos_encoder = joblib.load("hascosigner_encoder.pkl")
dep_encoder = joblib.load("HasDependents_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Numerical Inputs
    age = float(request.form["Age"])
    income = float(request.form["Income"])
    loan_amount = float(request.form["LoanAmount"])
    credit_score = float(request.form["CreditScore"])
    months_employed = float(request.form["MonthsEmployed"])
    num_credit_lines = float(request.form["NumCreditLines"])
    interest_rate = float(request.form["InterestRate"])
    loan_term = float(request.form["LoanTerm"])

    # Categorical Inputs
    education = request.form["Education"]
    employment = request.form["EmploymentType"]
    marital = request.form["MaritalStatus"]
    mortgage = request.form["HasMortgage"]
    purpose = request.form["LoanPurpose"]
    cosigner = request.form["HasCoSigner"]
    dependents = request.form["HasDependents"]

    # Encode Categories
    edu_encoded = edu_encoder.transform([education])[0]
    emp_encoded = emp_encoder.transform([employment])[0]
    mar_encoded = mar_encoder.transform([marital])[0]
    mort_encoded = mort_encoder.transform([mortgage])[0]
    purp_encoded = purp_encoder.transform([purpose])[0]
    cos_encoded = cos_encoder.transform([cosigner])[0]
    dep_encoded = dep_encoder.transform([dependents])[0]

    # Scale only numerical features
    numeric_features = [[
        age,
        income,
        loan_amount,
        credit_score,
        months_employed,
        num_credit_lines,
        interest_rate,
        loan_term
    ]]

    scaled_numeric = scaler.transform(numeric_features)
    
    # Create final feature vector (same order as training)
    final_features = [[
        scaled_numeric[0][0],   # Age
        scaled_numeric[0][1],   # Income
        scaled_numeric[0][2],   # LoanAmount
        scaled_numeric[0][3],   # CreditScore
        scaled_numeric[0][4],   # MonthsEmployed
        scaled_numeric[0][5],   # NumCreditLines
        scaled_numeric[0][6],   # InterestRate
        scaled_numeric[0][7],   # LoanTerm
        edu_encoded,
        emp_encoded,
        mar_encoded,
        mort_encoded,
        purp_encoded,
        cos_encoded,
        dep_encoded
    ]]

    # Prediction
    prediction = int(model.predict(final_features)[0])

    # Probability (if supported)
    probability = None
    if hasattr(model, "predict_proba"):
        probability = round(model.predict_proba(final_features)[0][1] * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=False)