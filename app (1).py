
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Delivery Prediction System",
    page_icon="🚚",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR

# ==========================================
# 2. LOAD TRAINED MODELS AND DATA
# ==========================================

@st.cache_resource
def load_models():

    classification_model = joblib.load(
        MODEL_DIR / "classification_model.pkl"
    )

    regression_model = joblib.load(
        MODEL_DIR / "regression_model.pkl"
    )

    return classification_model, regression_model


@st.cache_data
def load_data():

    with open(MODEL_DIR / "evaluation.json", "r") as file:
        evaluation = json.load(file)

    df = pd.read_csv(BASE_DIR / "delivery_dataset.csv")

    return evaluation, df


try:
    classification_model, regression_model = load_models()
    evaluation, df = load_data()

except Exception as e:
    st.error(f"Application Loading Error: {e}")
    st.info(
        "Check that your models folder, model files, "
        "evaluation.json, and delivery_dataset.csv are uploaded."
    )
    st.stop()


# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🚚 Smart Delivery System")
st.sidebar.write("Machine Learning Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home Dashboard",
        "Dataset Overview",
        "Order Status Prediction",
        "Delivery Time Prediction",
        "Model Evaluation"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Smart Delivery Time and Order Status Prediction System"
)


# ==========================================
# 4. INPUT FUNCTION
# ==========================================

def delivery_inputs(prefix):

    col1, col2 = st.columns(2)

    with col1:

        distance = st.number_input(
            "Delivery Distance (km)",
            min_value=1.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
            key=f"{prefix}_distance"
        )

        traffic = st.selectbox(
            "Traffic Level",
            ["Low", "Medium", "High"],
            key=f"{prefix}_traffic"
        )

        weather = st.selectbox(
            "Weather Condition",
            ["Sunny", "Cloudy", "Rainy"],
            key=f"{prefix}_weather"
        )

        delivery_type = st.selectbox(
            "Delivery Type",
            ["Motorcycle", "Bicycle", "Car"],
            key=f"{prefix}_type"
        )

    with col2:

        preparation_time = st.number_input(
            "Preparation Time (minutes)",
            min_value=1,
            max_value=180,
            value=15,
            key=f"{prefix}_preparation"
        )

        order_hour = st.slider(
            "Order Hour",
            min_value=0,
            max_value=23,
            value=12,
            key=f"{prefix}_hour"
        )

        promised_delivery_time = st.number_input(
            "Promised Delivery Time (minutes)",
            min_value=5,
            max_value=300,
            value=45,
            key=f"{prefix}_promised"
        )

    input_data = pd.DataFrame([{
        "distance": distance,
        "traffic": traffic,
        "weather": weather,
        "delivery_type": delivery_type,
        "preparation_time": preparation_time,
        "order_hour": order_hour,
        "promised_delivery_time": promised_delivery_time
    }])

    return input_data


# ==========================================
# 5. HOME DASHBOARD
# ==========================================

if page == "Home Dashboard":

    st.title("🚚 Smart Delivery Prediction System")

    st.write(
        "Predict delivery time and identify whether "
        "an order will arrive on time or be delayed."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Delivery Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Machine Learning Models",
        "2"
    )

    col3.metric(
        "Input Features",
        "7"
    )

    st.divider()

    st.subheader("Project Overview")

    st.write("""
    This application uses supervised machine learning
    to analyze delivery information and generate predictions.

    **Classification Model**
    - Predicts whether an order will be On Time or Delayed.

    **Regression Model**
    - Estimates the actual delivery time in minutes.
    """)

    st.subheader("Available Prediction Models")

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
        ### 📦 Order Status Classification

        Predict the delivery status of an order.

        **Output:**
        - On Time
        - Delayed
        """)

    with col2:

        st.success("""
        ### ⏱️ Delivery Time Regression

        Estimate the delivery completion time.

        **Output:**
        - Predicted minutes
        """)

    st.divider()

    st.subheader("Order Status Summary")

    if "order_status" in df.columns:

        status_counts = df["order_status"].value_counts()

        st.bar_chart(status_counts)

        st.dataframe(
            status_counts.rename("Number of Orders"),
            use_container_width=True
        )


# ==========================================
# 6. DATASET OVERVIEW
# ==========================================

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    st.write(
        "Explore the delivery records used to train "
        "the machine learning models."
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Missing Values")

    missing_values = df.isnull().sum()

    st.dataframe(
        missing_values.rename("Missing Values"),
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Order Status Distribution")

    status_counts = df["order_status"].value_counts()

    st.bar_chart(status_counts)

    st.subheader("Delivery Type Distribution")

    if "delivery_type" in df.columns:

        st.bar_chart(
            df["delivery_type"].value_counts()
        )

    st.subheader("Traffic Distribution")

    if "traffic" in df.columns:

        st.bar_chart(
            df["traffic"].value_counts()
        )


# ==========================================
# 7. CLASSIFICATION PAGE
# ==========================================

elif page == "Order Status Prediction":

    st.title("📦 Order Status Prediction")

    st.write(
        "Determine whether a delivery is expected "
        "to arrive On Time or Delayed."
    )

    st.divider()

    input_data = delivery_inputs("classification")

    if st.button(
        "Predict Order Status",
        use_container_width=True
    ):

        try:

            prediction = classification_model.predict(
                input_data
            )[0]

            st.subheader("Prediction Result")

            status = str(prediction).strip().lower()

            if status in ["on time", "ontime", "on_time"]:

                st.success(
                    "✅ The order is predicted to arrive ON TIME."
                )

            else:

                st.error(
                    "⚠️ The order is predicted to be DELAYED."
                )

            if hasattr(classification_model, "predict_proba"):

                probabilities = classification_model.predict_proba(
                    input_data
                )[0]

                classes = classification_model.classes_

                st.subheader("Prediction Confidence")

                for label, probability in zip(classes, probabilities):

                    st.write(
                        f"{label}: {probability * 100:.2f}%"
                    )

                    st.progress(float(probability))

        except Exception as e:

            st.error(f"Classification Error: {e}")


# ==========================================
# 8. REGRESSION PAGE
# ==========================================

elif page == "Delivery Time Prediction":

    st.title("⏱️ Delivery Time Prediction")

    st.write(
        "Estimate the number of minutes required "
        "to complete a delivery."
    )

    st.divider()

    input_data = delivery_inputs("regression")

    if st.button(
        "Predict Delivery Time",
        use_container_width=True
    ):

        try:

            prediction = regression_model.predict(
                input_data
            )[0]

            st.subheader("Estimated Delivery Time")

            st.success(
                f"Predicted Delivery Time: {prediction:.2f} minutes"
            )

            st.metric(
                "Estimated Minutes",
                f"{prediction:.2f}"
            )

            st.info(
                f"Approximately {round(prediction)} minutes."
            )

        except Exception as e:

            st.error(f"Regression Error: {e}")


# ==========================================
# 9. MODEL EVALUATION PAGE
# ==========================================

elif page == "Model Evaluation":

    st.title("📈 Model Evaluation")

    # CLASSIFICATION RESULTS

    st.header("Classification Model")

    classification = evaluation["classification"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{classification['accuracy']:.2%}"
    )

    col2.metric(
        "Precision",
        f"{classification['precision']:.2%}"
    )

    col3.metric(
        "Recall",
        f"{classification['recall']:.2%}"
    )

    col4.metric(
        "F1-Score",
        f"{classification['f1_score']:.2%}"
    )

    st.subheader("Confusion Matrix")

    cm = classification["confusion_matrix"]

    confusion_df = pd.DataFrame(
        cm,
        index=["Actual On Time", "Actual Delayed"],
        columns=["Predicted On Time", "Predicted Delayed"]
    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )

    st.caption(
        "The confusion matrix compares actual and predicted order statuses."
    )

    # REGRESSION RESULTS

    st.divider()

    st.header("Regression Model")

    regression = evaluation["regression"]

    col1, col2 = st.columns(2)

    col1.metric(
        "MAE",
        f"{regression['mae']:.2f} minutes"
    )

    col2.metric(
        "MSE",
        f"{regression['mse']:.2f}"
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "RMSE",
        f"{regression['rmse']:.2f} minutes"
    )

    col4.metric(
        "R² Score",
        f"{regression['r2']:.4f}"
    )

    st.success(
        "Both machine learning models have been evaluated."
    )


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.caption(
    "Smart Delivery Time and Order Status Prediction System | "
    "Machine Learning Laboratory Activity"
)
