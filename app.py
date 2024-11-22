import pickle
import streamlit as st
import numpy as np

model_logistic = pickle.load(open('model_diabetes.sav', 'rb'))
model_random_forest = pickle.load(open('model_diabetes_random_forest.sav', 'rb'))
model_gaussian = pickle.load(open('model_diabetes_gaussian.sav', 'rb'))

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #fdfbfb, #ebedee);
            font-family: 'Roboto', sans-serif;
        }
        .main-header {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 20px;
            margin-bottom: 10px;
            text-shadow: 1px 1px 2px #bdc3c7;
        }
        .sub-header {
            text-align: center;
            color: #34495e;
            font-size: 1.2rem;
            margin-bottom: 30px;
        }
        .stButton > button {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: bold;
            box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }
        .stButton > button:hover {
            background-color: #2980b9;
        }
        .result-box {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
        }
        .result-positive {
            background-color: #e74c3c;
        }
        .result-negative {
            background-color: #27ae60;
        }
        .stRadio div {
            font-size: 1rem;
            font-weight: bold;
            color: #34495e;
            margin-bottom: 10px;
        }
        .stRadio div label {
            cursor: pointer;
            border: 2px solid #bdc3c7;
            border-radius: 5px;
            padding: 10px 15px;
            display: inline-block;
            margin: 5px;
            transition: all 0.3s ease;
        }
        .stRadio div label:hover {
            background-color: #ecf0f1;
            border-color: #3498db;
        }
        .stRadio div label input:checked + div {
            background-color: #3498db;
            color: white;
            border-color: #2980b9;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>Diabetes Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Fill in the details below to check for diabetes risk.</div>", unsafe_allow_html=True)

model_choice = st.sidebar.radio(
    "Choose Prediction Model:",
    options=["Logistic Regression", "Random Forest", "Gaussian Naive Bayes"],
    help="""
    - Logistic Regression: A linear model best for simpler datasets.
    - Random Forest: An ensemble model, ideal for complex patterns.
    - Gaussian Naive Bayes: Probabilistic model assuming feature independence.
    """
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, step=1, help="Number of times pregnant")
        Glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=120, help="Plasma glucose concentration")
        BloodPressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, value=80, help="Diastolic blood pressure")
        SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, help="Triceps skinfold thickness")
    with col2:
        Insulin = st.number_input('Insulin Level (μU/mL)', min_value=0, max_value=1000, value=30, help="2-Hour serum insulin")
        BMI = st.number_input('BMI', min_value=0.0, max_value=60.0, value=22.0, format="%.1f", help="Body Mass Index")
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, format="%.2f", help="Family history of diabetes")
        Age = st.number_input('Age', min_value=1, max_value=120, value=25, help="Age in years")
    submitted = st.form_submit_button("Predict Diabetes")

if submitted:
    model = model_logistic if model_choice == 'Logistic Regression' else model_random_forest if model_choice == 'Random Forest' else model_gaussian
    features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    diabetes_prediction = model.predict(features)

    if diabetes_prediction[0] == 1:
        st.markdown("<div class='result-box result-positive'>The patient is likely to have diabetes.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box result-negative'>The patient is unlikely to have diabetes.</div>", unsafe_allow_html=True)
