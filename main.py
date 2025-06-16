import streamlit as st
import joblib
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json

# Load animation from local file or URL
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Load models
# predictor = joblib.load('./Models/predictor.pkl')
# risk_model = joblib.load('./Models/risk.pkl')
# stage_model = joblib.load('./Models/stage.pkl')

try:
    predictor = joblib.load('./Models/predictor.pkl')
    risk_model = joblib.load('./Models/risk.pkl')
    stage_model = joblib.load('./Models/stage.pkl')
except Exception as e:
    st.error(f"Failed to load models: {e}")


st.set_page_config(page_title="Lungs Cancer Prediction Dashboard", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910768.png", width=100)
    st.title("Navigation")
    selection = option_menu(menu_title=None,
                            options=["Home", "Predict Cancer"],
                            icons=["house", "activity"],
                            menu_icon="cast",
                            default_index=1)

if selection == "Home":
    st.title("Welcome to the Cancer Prediction System")
    colA1, colA2 = st.columns([1, 2])
    with colA1:
        st_lottie(load_lottie("Animation.json"), speed=1, loop=True, quality="high", height=400)
    with colA2:
        st.markdown("""
            ### About
            This dashboard provides:
            - **Cancer presence detection**
            - **Cancer risk score prediction** (scaled to 0–100)
            - **Stage of cancer prediction**

            It uses trained machine learning models for real-time inference.
        """)

elif selection == "Predict Cancer":
    st.title('🩺 Cancer Prediction System')

    with st.expander("➕ Enter Patient Details"):
        all_yes = st.toggle("Set all to 'Yes'?")
        default = 'Yes' if all_yes else 'No'
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ['Male', 'Female'])
            age = st.number_input("Age", 1, 120, 50)
            smoking = st.selectbox("Smoking", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            yellow_fingers = st.selectbox("Yellow Fingers", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            anxiety = st.selectbox("Anxiety", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            peer_pressure = st.selectbox("Peer Pressure", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            chronic_disease = st.selectbox("Chronic Disease", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            fatigue = st.selectbox("Fatigue", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
        with col2:
            allergy = st.selectbox("Allergy", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            wheezing = st.selectbox("Wheezing", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            alcohol = st.selectbox("Alcohol Consumption", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            coughing = st.selectbox("Coughing", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            shortness_breath = st.selectbox("Shortness of Breath", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            swallowing_diff = st.selectbox("Swallowing Difficulty", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            chest_pain = st.selectbox("Chest Pain", ['Yes', 'No'], index=0 if default == 'Yes' else 1)
            days_to_cancer = st.number_input("Days to Cancer Diagnosis", min_value=0, value=0)


        st.markdown("**Race (select all that apply):**")
        race_options = [
            'Asian',
            'White',
            'More than one race',
            'Black or African-American',
            'American Indian or Alaskan Native',
            'Native Hawaiian or Other Pacific Islander',
            'Participant refused to answer',
            'Others'
        ]  
        race_selected = st.radio("Select Race", race_options)

        # Set race variables based on selection
        race_american = race_selected == 'American Indian or Alaskan Native'
        race_asian = race_selected == 'Asian'
        race_black = race_selected == 'Black or African-American'
        race_more = race_selected == 'More than one race'
        race_hawaiian = race_selected == 'Native Hawaiian or Other Pacific Islander'
        race_others = race_selected == 'Others'
        race_refused = race_selected == 'Participant refused to answer'
        race_white = race_selected == 'White'

    def get_presence_input():
        return np.array([
            1 if gender == 'Male' else 0,
            age,
            1 if smoking == 'Yes' else 0,
            1 if yellow_fingers == 'Yes' else 0,
            1 if anxiety == 'Yes' else 0,
            1 if peer_pressure == 'Yes' else 0,
            1 if chronic_disease == 'Yes' else 0,
            1 if fatigue == 'Yes' else 0,
            1 if allergy == 'Yes' else 0,
            1 if wheezing == 'Yes' else 0,
            1 if alcohol == 'Yes' else 0,
            1 if coughing == 'Yes' else 0,
            1 if shortness_breath == 'Yes' else 0,
            1 if swallowing_diff == 'Yes' else 0,
            1 if chest_pain == 'Yes' else 0
        ]).reshape(1, -1)

    def get_risk_stage_input():
        return np.array([
            age,
            1 if gender == 'Male' else 0,
            days_to_cancer,
            int(race_american), int(race_asian), int(race_black), int(race_more),
            int(race_hawaiian), int(race_others), int(race_refused), int(race_white)
        ]).reshape(1, -1)

    if st.button('🚀 Predict'):
        presence_input = get_presence_input()
        risk_stage_input = get_risk_stage_input()

        # Predictions
        presence = predictor.predict(presence_input)[0]

        stage = stage_model.predict(risk_stage_input)[0]

        risk_pred = risk_model.predict(risk_stage_input)[0]
        min_y, max_y = 0, 200
        risk_score = 100 * (risk_pred - min_y) / (max_y - min_y)
        risk_score = np.clip(risk_score, 0, 100)

        

        colR1, colR2 = st.columns(2)
        with colR1:
            st.metric(label="Risk Score (0-100)", value=f"{risk_score:.2f}%", delta=None)
        with colR2:
            if presence:
                st.metric(label="Predicted Stage", value=f"Stage {stage}")
            else:
                st.metric(label="Predicted Stage", value="No stage")
        
        # Display results
        if presence:
            st.success("⚠️ Cancer Detected. Please consult a medical professional.")
        else:
            st.info("✅ No cancer detected based on current data.")
