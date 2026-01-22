import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

# Load Model
@st.cache_resource
def load_titanic_model():
    model_path = os.path.join('model', 'titanic_survival_model.pkl')
    return joblib.load(model_path)

try:
    model = load_titanic_model()
except Exception as e:
    st.error("Model file not found. Please run the model_building.ipynb first.")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to see if they would have survived the disaster.")

# User Inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Ticket Class (1st, 2nd, 3rd)", [1, 2, 3])
        sex = st.radio("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=1, max_value=100, value=30)
        
    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare Paid (£)", min_value=0.0, value=32.0)

    submit = st.form_submit_button("Predict Survival")

if submit:
    # Convert Sex to numeric (matching training logic)
    sex_numeric = 1 if sex == "Female" else 0
    
    # Arrange features
    input_data = np.array([[pclass, sex_numeric, age, sibsp, fare]])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("### Result: The passenger would have **SURVIVED**.")
        st.balloons()
    else:
        st.error("### Result: The passenger would **NOT** have survived.")