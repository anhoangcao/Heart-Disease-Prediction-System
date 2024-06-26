import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle
from pymongo import MongoClient
from urllib.parse import quote_plus
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# MongoDB setup - Should replace with st.secrets for production
username = "anhoang100402"
password = "Abc1234"
username_encoded = quote_plus(username)
password_encoded = quote_plus(password)
mongo_uri = f"mongodb+srv://{username_encoded}:{password_encoded}@cluster0.qeoxq3z.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(mongo_uri)
db = client['user_db']  # The database name
collection = db['information_heart_keys']  # The collection name

# Paths to your dataset and model
DATASET_PATH = "data/heart_2020_cleaned.csv"
LOG_MODEL_PATH = "model/voting_classifier.pkl"

def main():
    @st.cache(persist=True)
    def load_dataset() -> pd.DataFrame:
        heart_df = pl.read_csv(DATASET_PATH).to_pandas()
        heart_df = pd.DataFrame(np.sort(heart_df.values, axis=0),
                                index=heart_df.index,
                                columns=heart_df.columns)
        return heart_df

    # Add an input field for the username at the top of the sidebar
    st.sidebar.title("Patient Name")
    username_patient = st.sidebar.text_input("Enter your name", "")

    def user_input_features() -> pd.DataFrame:
        race = st.sidebar.selectbox("Chủng tộc", options=(race for race in heart.Race.unique()))
        sex = st.sidebar.selectbox("Giới tính", options=(sex for sex in heart.Sex.unique()))
        age_cat = st.sidebar.selectbox("Độ tuổi",
                                       options=(age_cat for age_cat in heart.AgeCategory.unique()))
        bmi_cat = st.sidebar.selectbox("Chỉ số BMI",
                                       options=(bmi_cat for bmi_cat in heart.BMICategory.unique()))
        sleep_time = st.sidebar.number_input("Trung bình bạn ngủ bao nhiêu giờ?", 0, 24, 7)
        gen_health = st.sidebar.selectbox("Tình trạng sức khỏe hiện tại của bạn như thế nào?",
                                          options=(gen_health for gen_health in heart.GenHealth.unique()))
        phys_health = st.sidebar.number_input("Có bao nhiêu ngày trong 30 ngày qua"
                                              " sức khỏe thể chất của bạn không tốt?", 0, 30, 0)
        ment_health = st.sidebar.number_input("Có bao nhiêu ngày trong 30 ngày qua"
                                              " sức khỏe tinh thần của bạn không tốt?", 0, 30, 0)
        phys_act = st.sidebar.selectbox("Bạn có chơi môn thể thao nào (chạy, đạp xe, v.v.)"
                                        " trong tháng qua không?", options=("No", "Yes"))
        smoking = st.sidebar.selectbox("Bạn đã hút ít nhất 100 điếu thuốc trong đời (khoảng 5 gói) chưa",
                                       options=("No", "Yes"))
        alcohol_drink = st.sidebar.selectbox("Bạn có uống hơn 14 ly rượu (nam) hoặc hơn 7 ly (nữ) trong một tuần không?", options=("No", "Yes"))
        stroke = st.sidebar.selectbox("Bạn có bị đột quỵ không?", options=("No", "Yes"))
        diff_walk = st.sidebar.selectbox("Bạn có gặp khó khăn nghiêm trọng khi đi bộ hoặc leo cầu thang không?", options=("No", "Yes"))
        diabetic = st.sidebar.selectbox("Bạn đã bao giờ mắc bệnh tiểu đường chưa?",
                                        options=(diabetic for diabetic in heart.Diabetic.unique()))
        asthma = st.sidebar.selectbox("Bạn có bị hen suyễn không?", options=("No", "Yes"))
        kid_dis = st.sidebar.selectbox("Bạn có bị bệnh thận không?", options=("No", "Yes"))
        skin_canc = st.sidebar.selectbox("Bạn có bị ung thư da không?", options=("No", "Yes"))

        features = pd.DataFrame({
            "PhysicalHealth": [phys_health],
            "MentalHealth": [ment_health],
            "SleepTime": [sleep_time],
            "BMICategory": [bmi_cat],
            "Smoking": [smoking],
            "AlcoholDrinking": [alcohol_drink],
            "Stroke": [stroke],
            "DiffWalking": [diff_walk],
            "Sex": [sex],
            "AgeCategory": [age_cat],
            "Race": [race],
            "Diabetic": [diabetic],
            "PhysicalActivity": [phys_act],
            "GenHealth": [gen_health],
            "Asthma": [asthma],
            "KidneyDisease": [kid_dis],
            "SkinCancer": [skin_canc]
        })

        return features

    # Create a row for the logout button, adjusting columns for alignment
    _, right_col = st.columns([0.8, 0.2])  # Adjust the ratio as needed
    # Layout for logout button at the top right
    st.title("Diagnosis is by signs of heart disease")
    
    with right_col:
        if st.button("Logout"):
            # Clear specific session state related to user session
            for key in ["login_status", "username", "account_type"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Set a flag to indicate user has logged out
            st.session_state['logged_out'] = True
            # Redirect to the main page by using st.experimental_rerun()
            st.experimental_rerun()
            
    st.subheader("Are you wondering about the condition of your heart? "
                 "This app will help you to diagnose it!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/doctor.png",
                 caption="I'll help you diagnose your heart health! - Dr. Voting Classifier",
                 width=150)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
        Did you know that machine learning models can help you
        predict heart disease pretty accurately? In this app, you can
        estimate your chance of heart disease (yes/no) in seconds!
        
        Here, a Voting Classifier model using an underselling technique
        was constructed using survey data of over 300k US residents from the year 2020.
        This application is based on it because it has proven to be better than the random forest
        (it achieves an accuracy of about 80%, which is quite good).
        
        To predict your heart disease status, simply follow the steps bellow:
        1. Enter the parameters that best describe you;
        2. Press the "Predict" button and wait for the result.
            
        **Keep in mind that this results is not equivalent to a medical diagnosis!
        This model would never be adopted by health care facilities because of its less
        than perfect accuracy, so if you have any problems, consult a human doctor.**
        
        **Author: Hoang An ([GitHub](https://github.com/anhoangcao))**
                    
        You can see the steps of building the model, evaluating it, and cleaning the data itself
        on my GitHub repo [here](https://github.com/anhoangcao/Heart-Disease-Prediction-System).
         
        """)

    heart = load_dataset()

    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/heart-sidebar.png", width=100)

    input_df = user_input_features()
    df = pd.concat([input_df, heart], axis=0)
    df = df.drop(columns=["HeartDisease"])

    cat_cols = ["BMICategory", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity",
                "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
    for cat_col in cat_cols:
        dummy_col = pd.get_dummies(df[cat_col], prefix=cat_col)
        df = pd.concat([df, dummy_col], axis=1)
        del df[cat_col]

    df = df[:1]
    df.fillna(0, inplace=True)

    log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

    if submit:
        # Assuming 'df' is your input DataFrame for the model
        prediction = log_model.predict(df)
        prediction_prob = log_model.predict_proba(df)
        
        # Displaying results based on the prediction
        if prediction == 0:
            # Corrected to show the probability of NOT having heart disease for clarity
            st.success(f"Kết quả dự đoán: No Heart Disease")
            st.markdown(f"**Xác suất bạn bị bệnh tim là {round(prediction_prob[0][1] * 100, 2)}%. Bạn khỏe mạnh!**")
            st.image("images/heart-okay.jpg", caption="Your heart seems to be okay! - Dr. Voting Classifier")
        else:
            # Keeps the probability of having heart disease
            st.success(f"Kết quả dự đoán: Heart Disease")
            st.markdown(f"**Xác suất bạn bị bệnh tim là {round(prediction_prob[0][1] * 100, 2)}%. Có vẻ như bạn không được khỏe.**")
            st.image("images/heart-bad.jpg", caption="I'm not satisfied with the condition of your heart! - Dr. Voting Classifier")

        record = {
            "username": username_patient,  # Assuming 'username_patient' holds the patient's username
            "user_input": input_df.iloc[0].to_dict(),  # 'input_df' should have been defined with the user input data
            "prediction": int(prediction[0]),
            # Corrected to store both probabilities in a list or dict format for clarity
            "prediction_probability": prediction_prob[0][1],
            "timestamp": datetime.now()  # Captures the current timestamp
        }

        
        # Insert the record into MongoDB
        collection.insert_one(record)
        
        st.write("Prediction saved successfully!")

        
if __name__ == "__main__":
    main()
