# app_main.py
import streamlit as st
import app_heart_key
import app_heart_sound

# Set page config in the main app only
st.set_page_config(
        page_title="Heart Disease Prediction App",
        page_icon="images/heart-fav.png"
)
def run_app_heart_key():
    app_heart_key.main()

def run_app_heart_sound():
    app_heart_sound.main()

st.title('Heart Condition Prediction')

tab1, tab2 = st.tabs(["Predict by Key Indicators of Heart Disease", "Predict by Heart Sound"])
with tab1:
    run_app_heart_key()
with tab2:
    run_app_heart_sound()

