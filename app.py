import streamlit as st
import pandas as pd
import joblib


# Load model
model = joblib.load("telco_churn_model.pkl")


# Page settings
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# Title
st.title("Telco Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")


st.divider()


# Customer Information

st.subheader("Customer Information")

col1, col2 = st.columns(2)


with col1:

    senior_citizen = st.selectbox(
        "Senior Citizen (+65 → 1)",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    # Multiple Lines depends on Phone Service
    if phone_service == "No":

        multiple_lines = "No"

        st.selectbox(
            "Multiple Lines",
            ["No"],
            disabled=True
        )

    else:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes"]
        )


with col2:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    # Internet-related services depend on Internet Service
    if internet_service == "No":

        online_security = "No"
        st.selectbox(
            "Online Security",
            ["No"],
            disabled=True
        )

        online_backup = "No"
        st.selectbox(
            "Online Backup",
            ["No"],
            disabled=True
        )

        device_protection = "No"
        st.selectbox(
            "Device Protection",
            ["No"],
            disabled=True
        )

        tech_support = "No"
        st.selectbox(
            "Tech Support",
            ["No"],
            disabled=True
        )

        streaming_tv = "No"
        st.selectbox(
            "Streaming TV",
            ["No"],
            disabled=True
        )

        streaming_movies = "No"
        st.selectbox(
            "Streaming Movies",
            ["No"],
            disabled=True
        )

    else:

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes"]
        )


# Billing Information

st.subheader("Billing Information")

col1, col2 = st.columns(2)


with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )


with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )


st.divider()


# Prediction

if st.button("Predict Churn", use_container_width=True):

    customer = pd.DataFrame([{
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer)[0]

    if prediction == 1:

        st.error("The customer is likely to churn.")

    else:

        st.success("The customer is unlikely to churn.")
