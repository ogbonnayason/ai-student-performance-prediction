import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Set page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("This app predicts whether a student will pass or fail based on academic and social features.")

# Sidebar input form
def user_input_features():
    st.sidebar.header("Enter Student Information")
    
    sex_input = st.sidebar.selectbox("Sex", ['male', 'female'])
    age = st.sidebar.slider("Age", 15, 22, 17)
    studytime = st.sidebar.selectbox("Weekly Study Time", [1, 2, 3, 4])
    failures = st.sidebar.slider("Past Class Failures", 0, 4, 0)
    absences = st.sidebar.slider("Number of Absences", 0, 100, 4)
    internet_input = st.sidebar.selectbox("Internet Access at Home", ['yes', 'no'])
    famsup_input = st.sidebar.selectbox("Family Educational Support", ['yes', 'no'])
    goout = st.sidebar.slider("Going Out with Friends (1–5)", 1, 5, 3)
    G1 = st.sidebar.slider("First Period Grade (G1)", 0, 20, 12)
    G2 = st.sidebar.slider("Second Period Grade (G2)", 0, 20, 12)

    # Encode categorical variables
    sex = 1 if sex_input == 'male' else 0
    internet = 1 if internet_input == 'yes' else 0
    famsup = 1 if famsup_input == 'yes' else 0

    data = {
        'sex': sex,
        'age': age,
        'studytime': studytime,
        'failures': failures,
        'absences': absences,
        'internet': internet,
        'famsup': famsup,
        'goout': goout,
        'G1': G1,
        'G2': G2
    }

    return pd.DataFrame([data])

# Get input features
input_df = user_input_features()

# Model options
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Sidebar model selection
st.sidebar.markdown("## Model Selection")
model_choice = st.sidebar.selectbox("Select a Machine Learning Model", list(models.keys()))

# Load the corresponding model
model_filename = f"{model_choice.lower().replace(' ', '_')}_model.joblib"

try:
    model = load(model_filename)
    st.success(f"{model_choice} model loaded successfully.")
except FileNotFoundError:
    st.error(f"Model file '{model_filename}' not found. Please export your trained model as '{model_filename}'.")
    st.stop()

# Display user inputs
st.subheader("Student Information")
st.write(input_df)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.success("✅ The student is predicted to PASS.")
    else:
        st.warning("❌ The student is predicted to FAIL.")

    st.subheader("Prediction Probability")
    st.write(f"Probability of passing: {prediction_proba[1]*100:.2f}%")
    st.write(f"Probability of failing: {prediction_proba[0]*100:.2f}%")
