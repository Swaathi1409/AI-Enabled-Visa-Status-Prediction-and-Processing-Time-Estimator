# utils.py
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import streamlit as st

@st.cache_data(ttl=3600)
def load_reference_data(path="visa_dataset_feature_engineered.csv"):
    """Load the dataset used to compute averages / defaults."""
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_artifacts(prep_path="visa_preprocessor.pkl", model_path="best_regression_model.pkl", df_path="visa_dataset_feature_engineered.csv"):
    """Load preprocessor and model once per session."""
    preprocessor = joblib.load(prep_path)
    model = joblib.load(model_path)
    df = load_reference_data(df_path)
    return preprocessor, model, df

# helper functions: these must match Milestone 2 logic
def compute_date_features(dt):
    """Given date-like (string or datetime), return month, weekday, iso-week"""
    d = pd.to_datetime(dt)
    return int(d.month), int(d.weekday()), int(d.isocalendar().week)

def map_season_index(season):
    mapping = {"Low": 1, "Mid": 2, "Off-Peak": 3, "Peak": 4}
    return mapping.get(season, 2)

def get_country_avg(df, country):
    try:
        v = df.loc[df["Applicant Nationality"] == country, "Processing Time (Days)"].mean()
        return float(v) if not np.isnan(v) else float(df["Processing Time (Days)"].mean())
    except Exception:
        return float(df["Processing Time (Days)"].mean())

def get_visa_avg(df, visa_type):
    try:
        v = df.loc[df["Visa Type"] == visa_type, "Processing Time (Days)"].mean()
        return float(v) if not np.isnan(v) else float(df["Processing Time (Days)"].mean())
    except Exception:
        return float(df["Processing Time (Days)"].mean())

def get_center_load(df, center):
    try:
        return float(df["Processing Center"].value_counts().get(center, df.shape[0] / 10.0))
    except Exception:
        return float(df.shape[0] / 10.0)

def compute_prediction_interval(df_reference, residuals, alpha=0.05):
    """
    Simple approximation for prediction interval: use residual distribution (RMSE).
    Returns multiplier (z) for 95% ~1.96 by default for alpha=0.05.
    We will return RMSE to compute +/- later in app.
    """
    rmse = float(np.sqrt((residuals ** 2).mean()))
    return rmse

def build_input_row(df_reference, application_date, decision_date, visa_type,
                    nationality, center, season, complexity, completeness, expedited):
    # derive date features
    app_m, app_d, app_w = compute_date_features(application_date)
    dec_m, dec_d, dec_w = compute_date_features(decision_date)

    season_idx = map_season_index(season)
    country_avg = get_country_avg(df_reference, nationality)
    visa_avg = get_visa_avg(df_reference, visa_type)
    center_load = get_center_load(df_reference, center)

    row = pd.DataFrame([{
        "Application Date": str(application_date),
        "Decision Date": str(decision_date),
        "Visa Type": visa_type,
        "Applicant Nationality": nationality,
        "Processing Center": center,
        "Season": season,
        "Application Complexity": float(complexity),
        "Document Completeness": float(completeness),
        "Expedited Request": float(expedited),
        "Application_Month": int(app_m),
        "Application_DayOfWeek": int(app_d),
        "Application_WeekOfYear": int(app_w),
        "Decision_Month": int(dec_m),
        "Decision_DayOfWeek": int(dec_d),
        "Decision_WeekOfYear": int(dec_w),
        "Season_Index": int(season_idx),
        "Country_Avg_Processing": float(country_avg),
        "VisaType_Avg_Processing": float(visa_avg),
        "Center_Load": float(center_load)
    }])

    # If the preprocessor expects additional columns, ensure they exist here.
    return row

def predict_from_input(application_date, decision_date, visa_type,
                       nationality, center, season, complexity, completeness, expedited,
                       prep_path="visa_preprocessor.pkl", model_path="best_regression_model.pkl",
                       df_path="visa_dataset_feature_engineered.csv"):
    preprocessor, model, df_reference = load_artifacts(prep_path, model_path, df_path)

    row = build_input_row(df_reference, application_date, decision_date, visa_type,
                          nationality, center, season, complexity, completeness, expedited)

    # transform with preprocessor used in training
    X_trans = preprocessor.transform(row)
    pred = model.predict(X_trans)[0]

    rmse = None
    try:
        # try to compute RMSE using model on a subset (may be slow) - optional safe fallback
        sample_X = df_reference.drop(columns=["Processing Time (Days)", "Visa Status"], errors="ignore").iloc[:1000]
        sample_X_trans = preprocessor.transform(sample_X)
        sample_y = df_reference["Processing Time (Days)"].iloc[:1000]
        preds_sample = model.predict(sample_X_trans)
        rmse = float(np.sqrt(((sample_y - preds_sample) ** 2).mean()))
    except Exception:
        rmse = float(df_reference["Processing Time (Days)"].std())

    return float(max(0.0, pred)), rmse
