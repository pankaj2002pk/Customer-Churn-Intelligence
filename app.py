import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =============================================================================
# 1. PAGE CONFIGURATION & ENTERPRISE STYLING
# =============================================================================
st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 30px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 15px;
        color: #4B5563;
        margin-bottom: 22px;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. LOAD TRAINED MODEL ARTIFACTS
# =============================================================================
@st.cache_resource
def load_ml_pipeline():
    model = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/features_list.pkl')
    return model, scaler, features

try:
    model, scaler, expected_features = load_ml_pipeline()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure churn_model.pkl, scaler.pkl, and features_list.pkl exist inside the 'models' directory.")
    st.stop()

# =============================================================================
# 3. SIDEBAR: CUSTOMER PROFILE & SERVICE INPUTS
# =============================================================================
st.sidebar.header("📋 Customer Profile")

st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

st.sidebar.subheader("Subscribed Services")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.sidebar.subheader("Contract & Billing")
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

st.sidebar.subheader("Usage & Charges")
tenure = st.sidebar.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.5)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(round(tenure * monthly_charges, 2)), step=10.0)

# =============================================================================
# 4. MAIN DASHBOARD CONTENT
# =============================================================================
st.markdown('<div class="main-header">Telecom Customer Churn Predictor & Retention Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time enterprise inference pipeline powered by Balanced Logistic Regression (Optimal Recall ~78.3%).</div>', unsafe_allow_html=True)

# Overview KPI row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Customer Tenure", value=f"{tenure} Months")
with col2:
    st.metric(label="Contract Plan", value=contract)
with col3:
    st.metric(label="Monthly Charge", value=f"${monthly_charges:.2f}")
with col4:
    st.metric(label="Total Billed", value=f"${total_charges:.2f}")

st.write("---")

# =============================================================================
# 5. DATA INGESTION & PIPELINE PREPARATION
# =============================================================================
input_raw = {
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender': gender,
    'Partner': partner,
    'Dependents': dependents,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method
}

raw_df = pd.DataFrame([input_raw])

# One-hot encode inputs matching training drop_first strategy
encoded_df = pd.get_dummies(raw_df, drop_first=True)

# Strict schema reindexing: aligns columns exactly to the 30 training features
aligned_df = encoded_df.reindex(columns=expected_features, fill_value=0)

# Scale numeric columns with the fitted StandardScaler
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
aligned_df[numeric_features] = scaler.transform(aligned_df[numeric_features])

# Generate ML Prediction & Probability
prediction = model.predict(aligned_df)[0]
churn_prob = model.predict_proba(aligned_df)[0][1]

# =============================================================================
# 6. BUSINESS RESULTS & RETENTION STRATEGY CARDS (CORPORATE ENGLISH)
# =============================================================================
res_left, res_right = st.columns([1.2, 1])

with res_left:
    st.subheader("🎯 Model Prediction & Risk Score")
    
    if prediction == 1:
        st.error("⚠️ **HIGH CHURN RISK DETECTED**")
        st.markdown(f"The estimated probability of this customer discontinuing service is **{churn_prob * 100:.2f}%**.")
    else:
        st.success("✅ **LOW CHURN RISK (RETAINED CUSTOMER)**")
        st.markdown(f"High likelihood of customer retention. Estimated churn risk is only **{churn_prob * 100:.2f}%**.")
        
    st.progress(float(churn_prob))
    
    st.info(f"Inference Threshold: 50.0% | Customer Status: {'At-Risk (Churn)' if prediction == 1 else 'Active & Retained'}")

with res_right:
    st.subheader("💼 Recommended Business Strategy")
    if prediction == 1:
        st.warning("**Retention Strategy for Customer Success Team:**")
        st.markdown("""
        * **Long-Term Contract Incentive:** Provide targeted annual discounts to migrate customers from month-to-month to 1-year or 2-year agreements.
        * **Automated Payment Transition:** Offer recurring bill credits or incentives to switch from electronic check to automated bank transfer or credit card billing.
        * **Value-Added Service Bundling:** Attach complimentary technical support or premium device protection packages to improve customer stickiness.
        """)
    else:
        st.success("**Upselling & Engagement Strategy:**")
        st.markdown("""
        * **Loyalty & Tier Rewards:** Provide tenure-based milestone rewards and dedicated VIP customer support access.
        * **Cross-Sell Premium Services:** Recommend high-speed bandwidth upgrades, multi-device plans, and premium OTT content bundles.
        """)

st.write("---")
st.caption("Machine Learning Architecture: Balanced Logistic Regression | Target Recall: ~78.3% | Framework: Scikit-Learn & Streamlit")