
import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Smart Delivery Prediction",
    page_icon="🚚",
    layout="wide"
)

# Load Machine Learning Models
@st.cache_resource
def load_models():

    classification_model = joblib.load("delivery_classification.pkl")
    regression_model = joblib.load("delivery_regression.pkl")

    return classification_model, regression_model


try:
    classification_model, regression_model = load_models()

except Exception as e:
    st.error(f"Unable to load machine learning models: {e}")
    st.info(
        "Please check that both .pkl files are uploaded "
        "to your GitHub repository."
    )
    st.stop()


# Application Title
st.title("🚚 Smart Delivery Prediction System")

st.write(
    "Predict delivery status and estimated arrival time "
    "using Machine Learning."
)

# Sidebar Navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Delivery Prediction"]
)


# HOME PAGE
if page == "Home":

    st.title("Welcome to Smart Delivery Prediction!")

    st.write("""
    This system uses machine learning to help predict
    delivery delays and estimate delivery times.

    **Classification:** Predicts whether a delivery is
    On-Time or Delayed.

    **Regression:** Estimates delivery time in minutes.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Delivery Classification")
        st.write("Identifies whether a delivery will arrive on time or be delayed.")

    with col2:
        st.subheader("⏱️ Delivery Time Prediction")
        st.write("Estimates the expected delivery duration in minutes.")


# DELIVERY PREDICTION PAGE
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

    if st.button("🚚 Predict Delivery", use_container_width=True):

        data = pd.DataFrame({
            "Distance": [distance],
            "Traffic": [traffic],
            "Weather": [weather],
            "Preparation_Time": [preparation]
        })

        try:

            # Classification Prediction
            status = classification_model.predict(data)[0]

            # Regression Prediction
            time = regression_model.predict(data)[0]

            st.divider()
            st.subheader("Prediction Results")

            if status == "Delayed":
                st.warning("⚠️ Delivery Status: DELAYED")
            else:
                st.success("✅ Delivery Status: ON-TIME")

            st.metric(
                label="Estimated Delivery Time",
                value=f"{time:.1f} minutes"
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")


# FOOTER
st.sidebar.markdown("---")
st.sidebar.caption("Artificial Intelligence 5.0")
