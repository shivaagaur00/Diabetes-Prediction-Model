import pickle
import streamlit as st
import numpy as np

model_logistic = pickle.load(open('model_diabetes.sav', 'rb'))
model_random_forest = pickle.load(open('model_diabetes_random_forest.sav', 'rb'))
model_gaussian = pickle.load(open('model_diabetes_gaussian.sav', 'rb'))

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
            font-family: 'Roboto', sans-serif;
        }
        .main-header {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .sub-header {
            text-align: center;
            color: #34495e;
            font-size: 1.5rem;
            margin-bottom: 20px;
        }
        .stButton > button {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #2980b9;
        }
        .result-box {
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .result-positive {
            background-color: #ff6f61;
            color: white;
            font-size: 1.5rem;
        }
        .result-negative {
            background-color: #2ecc71;
            color: white;
            font-size: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>Diabetes Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Please fill in the details below to check for diabetes risk.</div>", unsafe_allow_html=True)

with st.container():
    model_choice = st.selectbox('Choose the Prediction Model:', ['Logistic Regression', 'Random Forest', 'Gaussian Naive Bayes'])

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input('Pregnancies:', min_value=0, max_value=20, value=0, step=1)
        Glucose = st.number_input('Glucose Level:', min_value=0, max_value=300, value=120)
        BloodPressure = st.number_input('Blood Pressure (mmHg):', min_value=0, max_value=200, value=80)
        SkinThickness = st.number_input('Skin Thickness (mm):', min_value=0, max_value=100, value=20)
    with col2:
        Insulin = st.number_input('Insulin Level (μU/mL):', min_value=0, max_value=1000, value=30)
        BMI = st.number_input('BMI:', min_value=0.0, max_value=60.0, value=22.0, format="%.1f")
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function:', min_value=0.0, max_value=2.5, value=0.5, format="%.2f")
        Age = st.number_input('Age:', min_value=1, max_value=120, value=25)

if st.button('Predict Diabetes'):
    if model_choice == 'Logistic Regression':
        model = model_logistic
    elif model_choice == 'Random Forest':
        model = model_random_forest
    else:
        model = model_gaussian

    features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    diabetes_prediction = model.predict(features)

    if diabetes_prediction[0] == 1:
        st.markdown("<div class='result-box result-positive'>The patient is likely to have diabetes.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box result-negative'>The patient is unlikely to have diabetes.</div>", unsafe_allow_html=True)
