from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
DEFAULT_MODEL_URL = (
    "https://github.com/b4iterdev/ml_customer_churn_pred/releases/download/"
    "v1.0.0/best_model.joblib"
)


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


def ensure_model_available(model_path: Path) -> tuple[bool, str]:
    if model_path.exists() and model_path.stat().st_size > 0:
        return True, "Model đã sẵn sàng trên server."

    model_url = os.getenv("MODEL_URL", DEFAULT_MODEL_URL)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(".joblib.tmp")

    try:
        response = requests.get(model_url, timeout=60, stream=True)
        response.raise_for_status()

        with temp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if temp_path.stat().st_size == 0:
            temp_path.unlink(missing_ok=True)
            return False, "Tải model thất bại: file rỗng."

        temp_path.replace(model_path)
        return True, f"Đã tải model từ release: {model_url}"
    except requests.RequestException as exc:
        temp_path.unlink(missing_ok=True)
        return False, f"Không thể tải model từ release: {exc}"


def build_input_dataframe() -> pd.DataFrame:
    st.subheader("Thông tin khách hàng")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], index=0)
        partner = st.selectbox("Partner", ["No", "Yes"], index=0)
        dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"], index=1)
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
            index=0,
        )

    with c2:
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"], index=0
        )
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
            index=0,
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
            index=0,
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
            index=0,
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
            index=0,
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
            index=0,
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
            index=0,
        )

    with c3:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"], index=0
        )
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=0,
        )
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.1,
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0,
            step=0.1,
        )

    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
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
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }
    return pd.DataFrame([payload])


def main() -> None:
    st.set_page_config(
        page_title="Telco Churn Predictor", page_icon="📉", layout="wide"
    )
    st.title("Telco Customer Churn Predictor")
    st.caption("Dự báo khả năng khách hàng rời bỏ dịch vụ viễn thông")

    with st.spinner("Đang kiểm tra model dự báo..."):
        model_ok, model_message = ensure_model_available(MODEL_PATH)

    if not model_ok:
        st.error(model_message)
        st.info(
            "Bạn có thể cấu hình URL release bằng biến môi trường `MODEL_URL` "
            "trên Streamlit Cloud."
        )
        return

    st.caption(model_message)

    model = load_model(MODEL_PATH)
    input_df = build_input_dataframe()

    if st.button("Dự báo Churn", type="primary"):
        proba = float(model.predict_proba(input_df)[0][1])
        pred = int(model.predict(input_df)[0])

        st.subheader("Kết quả dự báo")
        if pred == 1:
            st.error(
                f"Khách hàng có nguy cơ rời bỏ dịch vụ (Churn=Yes). Xác suất: {proba:.2%}"
            )
        else:
            st.success(
                f"Khách hàng có xu hướng ở lại (Churn=No). Xác suất rời bỏ: {proba:.2%}"
            )

        st.progress(min(max(proba, 0.0), 1.0), text=f"Churn probability: {proba:.2%}")

        with st.expander("Dữ liệu đầu vào dùng để dự báo"):
            st.dataframe(input_df, use_container_width=True)


if __name__ == "__main__":
    main()
