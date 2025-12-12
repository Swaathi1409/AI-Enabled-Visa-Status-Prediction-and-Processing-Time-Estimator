# app.py (Streamlit UI)
import streamlit as st
import pandas as pd
from utils import predict_from_input, load_reference_data, load_artifacts

st.set_page_config(page_title="Visa Processing Time Estimator", layout="centered", initial_sidebar_state="expanded")

st.title("AI — Visa Processing Time Estimator")
st.markdown("Enter visa application details and get a predicted processing time (days).")

# load reference data for dropdowns
df_ref = load_reference_data("visa_dataset_feature_engineered.csv")

# Sidebar / inputs
with st.sidebar:
    st.header("Input options")
    visa_types = sorted(df_ref["Visa Type"].dropna().unique().tolist())
    countries = sorted(df_ref["Applicant Nationality"].dropna().unique().tolist())
    centers = sorted(df_ref["Processing Center"].dropna().unique().tolist())
    seasons = sorted(df_ref["Season"].dropna().unique().tolist())

    st.write("Tip: select reasonable values from dropdowns for best prediction.")

# Main input form
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        application_date = st.date_input("Application Date")
        visa_type = st.selectbox("Visa Type", options=visa_types)
        season = st.selectbox("Season", options=seasons)
        completeness = st.selectbox("Document Completeness", [0,1])
    with col2:
        decision_date = st.date_input("Decision Date")
        nationality = st.selectbox("Applicant Nationality", options=countries)
        center = st.selectbox("Processing Center", options=centers)
        expedited = st.selectbox("Expedited Request", [0,1])

    complexity = st.slider("Application Complexity (0=low,1=medium,2=high)", min_value=0, max_value=2, value=0)
    submitted = st.form_submit_button("Estimate Processing Time")

if submitted:
    # basic validation
    if decision_date < application_date:
        st.error("Decision date cannot be earlier than application date.")
    else:
        with st.spinner("Predicting..."):
            pred, rmse = predict_from_input(
                application_date=str(application_date),
                decision_date=str(decision_date),
                visa_type=visa_type,
                nationality=nationality,
                center=center,
                season=season,
                complexity=complexity,
                completeness=completeness,
                expedited=expedited,
                prep_path="visa_preprocessor.pkl",
                model_path="best_regression_model.pkl",
                df_path="visa_dataset_feature_engineered.csv"
            )

        st.success(f"Estimated Processing Time: **{pred:.1f} days**")
        if rmse is not None:
            st.info(f"Model RMSE (approx): {rmse:.2f} days — use ±1×RMSE as a rough uncertainty range.")
            st.write(f"Estimated range (approx): **{max(0, pred-rmse):.1f} – {pred+rmse:.1f} days**")

        # small visual: bar comparing predicted vs dataset mean
        mean_days = df_ref["Processing Time (Days)"].mean()
        st.metric("Dataset mean processing time (days)", f"{mean_days:.1f}")
        st.bar_chart(pd.DataFrame({"Predicted": [pred], "Dataset mean": [mean_days]}))
