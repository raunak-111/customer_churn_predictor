from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Configure Streamlit app layout and title.
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Predictor")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
MODEL_PATH = PROJECT_ROOT / "model.pkl"
FEATURE_COLS_PATH = PROJECT_ROOT / "feature_cols.pkl"
FEATURE_PLOT_PATH = PROJECT_ROOT / "feature_importance.png"


@st.cache_resource
def load_model_artifacts():
    """Load model and feature schema artifacts generated during training."""
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, feature_cols


@st.cache_resource
def build_training_preprocessor():
    """Rebuild and fit the same preprocessing pipeline used during model training."""
    df = pd.read_csv(DATA_PATH)

    # Apply the same cleaning and target encoding as the notebook training flow.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    pipeline.fit(X_train, y_train)
    return pipeline


def build_user_input_row(
    tenure,
    monthly_charges,
    total_charges,
    contract,
    internet_service,
    payment_method,
    senior_citizen,
    paperless_billing,
    online_security,
    tech_support,
):
    """Build one raw input row with defaults for non-exposed training features."""
    if internet_service == "No":
        online_security_value = "No internet service"
        tech_support_value = "No internet service"
        online_backup_value = "No internet service"
        device_protection_value = "No internet service"
        streaming_tv_value = "No internet service"
        streaming_movies_value = "No internet service"
    else:
        online_security_value = online_security
        tech_support_value = tech_support
        online_backup_value = "No"
        device_protection_value = "No"
        streaming_tv_value = "No"
        streaming_movies_value = "No"

    # Features not exposed in UI are assigned stable defaults so schema stays complete.
    row = {
        "gender": "Female",
        "SeniorCitizen": int(senior_citizen),
        "Partner": "No",
        "Dependents": "No",
        "tenure": int(tenure),
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet_service,
        "OnlineSecurity": online_security_value,
        "OnlineBackup": online_backup_value,
        "DeviceProtection": device_protection_value,
        "TechSupport": tech_support_value,
        "StreamingTV": streaming_tv_value,
        "StreamingMovies": streaming_movies_value,
        "Contract": contract,
        "PaperlessBilling": "Yes" if paperless_billing else "No",
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }
    return pd.DataFrame([row])


# Show feature importance chart in sidebar.
st.sidebar.header("Model Insights")
if FEATURE_PLOT_PATH.exists():
    st.sidebar.image(str(FEATURE_PLOT_PATH), caption="Top 10 Feature Importances")
else:
    st.sidebar.info("Feature importance image not found.")

# Build the requested input controls.
st.subheader("Customer Inputs")
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("tenure", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, value=70.0, step=1.0)
    total_charges = st.number_input("TotalCharges", min_value=0.0, value=850.0, step=1.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])

with col2:
    payment_method = st.selectbox(
        "PaymentMethod",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    senior_citizen = st.checkbox("SeniorCitizen", value=False)
    paperless_billing = st.checkbox("PaperlessBilling", value=True)
    online_security = st.selectbox("OnlineSecurity", ["Yes", "No"])
    tech_support = st.selectbox("TechSupport", ["Yes", "No"])

# Load model and preprocessing helpers once.
model, feature_cols = load_model_artifacts()
preprocess_pipeline = build_training_preprocessor()

if st.button("Predict", type="primary"):
    raw_input_df = build_user_input_row(
        tenure=tenure,
        monthly_charges=monthly_charges,
        total_charges=total_charges,
        contract=contract,
        internet_service=internet_service,
        payment_method=payment_method,
        senior_citizen=senior_citizen,
        paperless_billing=paperless_billing,
        online_security=online_security,
        tech_support=tech_support,
    )

    transformed = preprocess_pipeline.transform(raw_input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed_cols = preprocess_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    model_input_df = pd.DataFrame(transformed, columns=transformed_cols)

    # Enforce exact feature ordering from training artifacts.
    model_input_df = model_input_df.reindex(columns=feature_cols, fill_value=0.0)

    # Model was trained on a numpy array, so infer with array format to avoid warnings.
    churn_probability = float(model.predict_proba(model_input_df.values)[0, 1])

    st.metric("Churn Probability", f"{churn_probability:.2%}")
    st.progress(int(churn_probability * 100), text=f"Risk Level: {churn_probability:.0%}")

    if churn_probability > 0.5:
        st.error("High churn risk predicted.")
    else:
        st.success("Low churn risk predicted.")
