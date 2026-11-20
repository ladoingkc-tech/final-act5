
import streamlit as st
import pandas as pd
import joblib

classification_model = joblib.load("delivery_classification.pkl")
regression_model = joblib.load("delivery_regression.pkl")

st.set_page_config(
    page_title="Smart Delivery Prediction",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Smart Delivery Prediction System")
st.write("Predict delivery status and estimated arrival time.")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Delivery Prediction"]
)

if page == "Home":

    st.title("Welcome to Smart Delivery Prediction!")

    st.write("""
    This system uses machine learning to help predict
    delivery delays and estimate delivery times.

    **Classification:** Predicts On-Time or Delayed.

    **Regression:** Estimates delivery time in minutes.
    """)

elif page == "Delivery Prediction":

    st.header("Enter Delivery Information")

    distance = st.number_input(
        "Delivery Distance (km)",
        min_value=0.1,
        value=10.0
    )

    traffic = st.selectbox(
        "Traffic Condition",
        ["Low", "Medium", "High"]
    )

    weather = st.selectbox(
        "Weather Condition",
        ["Sunny", "Cloudy", "Rainy"]
    )

    preparation = st.number_input(
        "Order Preparation Time (minutes)",
        min_value=0.0,
        value=15.0
    )

    if st.button("Predict Delivery"):

        data = pd.DataFrame({
            "Distance": [distance],
            "Traffic": [traffic],
            "Weather": [weather],
            "Preparation_Time": [preparation]
        })

        status = classification_model.predict(data)[0]

        time = regression_model.predict(data)[0]

        st.subheader("Prediction Results")

        if status == "Delayed":
            st.warning("Delivery Status: DELAYED")
        else:
            st.success("Delivery Status: ON-TIME")

        st.metric(
            "Estimated Delivery Time",
            f"{time:.1f} minutes"
        )

st.sidebar.markdown("---")
st.sidebar.caption("Artificial Intelligence 5.0")
