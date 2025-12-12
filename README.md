# AI Enabled Visa Status Prediction & Processing Time Estimator

## Live Demo
**Streamlit App:**  
https://ai-visa-status-prediction-and-processing-time-estimator.streamlit.app/

---

## Project Overview
This project delivers an AI-powered system that predicts **visa processing time (regression)** based on application details.  
A unified multi-country dataset is engineered, processed, modeled, and deployed as an interactive Streamlit application.

---

## Milestone 1 — Dataset Creation & Preprocessing
- Consolidated visa data from multiple countries into a single dataset.
- Cleaned and standardized column names, formats, and values.
- Extracted date-based features:
  - `Application_Month`, `Application_DayOfWeek`, `Application_WeekOfYear`
  - `Decision_Month`, `Decision_DayOfWeek`, `Decision_WeekOfYear`
- Defined the target variable:  
  **Processing Time (Days)**
- Categorized columns into:
  - **Categorical:** Visa Type, Applicant Nationality, Processing Center, Season  
  - **Numeric:** All engineered date & complexity-related fields
- Built a preprocessing pipeline using Scikit-Learn:
  - Missing value imputation (`SimpleImputer`)
  - One-hot encoding for categorical features
  - Standard scaling for numeric features
- Saved preprocessing artifact:  
  **`visa_preprocessor.pkl`**

---

## Milestone 2 — Exploratory Data Analysis (EDA)
Performed data exploration using histograms, boxplots, and correlation heatmaps.

### Distribution Insights
- Histograms for numeric variables  
- Count plots for categorical fields  

### Relationship Insights
- Boxplots showing processing time variation across:
  - Visa Type  
  - Nationality  
  - Processing Center  
  - Season  
- Correlation matrix to understand numeric influence  
- Time-based pattern analysis across months and weeks  

---

## Feature Engineering
Added ML-optimized features to improve model accuracy:

- **Season_Index**  
  Numerical encoding of seasons (1–4)

- **Country_Avg_Processing**  
  Average processing time grouped by nationality

- **VisaType_Avg_Processing**  
  Average duration per visa category

- **Center_Load**  
  Approximate workload of each processing center

Final engineered dataset saved as:  
**`visa_dataset_feature_engineered.csv`**

---

## Milestone 3 — Regression Model Development & Training

### Algorithms Evaluated
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor 

Evaluation metrics used:
- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² Score  

### Final Output Artifact
The best-performing regression model saved as:  
**`best_regression_model.pkl`**

---

## Milestone 4 — Deployment (Streamlit Application)

### Features of the Streamlit App
- Accepts visa application details from user input  
- Predicts:
  - **Processing Time (Days)**  
  - **Optional prediction interval using RMSE**
- Uses previously saved artifacts:
  - `visa_preprocessor.pkl`
  - `best_regression_model.pkl`
  - `visa_dataset_feature_engineered.csv`
- Performs real-time inference in a clean UI

### Deployment
- Hosted on Streamlit Community Cloud  
- Fully integrated preprocessing + prediction pipeline

🔗 **Live App:**  
https://ai-visa-status-prediction-and-processing-time-estimator.streamlit.app/

---

## Screenshots

<img width="1166" height="670" alt="image" src="https://github.com/user-attachments/assets/ea55694c-df71-4f12-8312-6adc751b3e7e" />

<img width="504" height="498" alt="image" src="https://github.com/user-attachments/assets/4379d5ed-2ef4-4823-9db6-03e52def529d" />

---

## Technologies Used
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn  
- Streamlit  
- Joblib  

---

## License
MIT License

---

## Author
**Swaathi B**  
GitHub: **Swaathi1409**
