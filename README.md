# AI Enabled Visa Status Prediction & Processing Time Estimator

## Overview
This project develops an AI-powered system capable of predicting visa processing time (regression) and visa status (classification). It uses a combined multi-country dataset, performs complete preprocessing, EDA, feature engineering, and prepares the data for model training and deployment.

## Milestone 1 – Dataset Creation & Preprocessing
- Combined and cleaned multi-country visa datasets.
- Added date-based ML features:
  - Application_Month, Application_DayOfWeek, Application_WeekOfYear  
  - Decision_Month, Decision_DayOfWeek, Decision_WeekOfYear  
- Defined target variables:
  - **Processing Time (Days)**  
  - **Visa Status**
- Identified column types:
  - Categorical: Visa Type, Applicant Nationality, Processing Center, Season  
  - Numeric: All engineered numeric fields
- Built Scikit-Learn preprocessing pipeline:
  - Missing value handling (SimpleImputer)
  - One-hot encoding for categorical columns
  - Standard scaling for numeric columns
- Saved pipeline as `visa_preprocessor.pkl` for inference use.

## Milestone 2 – Exploratory Data Analysis (EDA)
Performed deep visual and statistical analysis using Matplotlib and Seaborn:

### Distribution Analysis
- Histograms for numeric features  
- Count plots for categorical features  

### Relationship Analysis
- Boxplots showing Processing Time variations across:
  - Visa Type  
  - Nationality  
  - Processing Center  
  - Season  
- Correlation matrix and heatmap for numeric feature influence  
- Trend patterns across month and week numbers  

## Feature Engineering
Additional ML-optimized features were created:

- **Season_Index**  
  Encodes seasonal trends:  
  1 = Apr–Jun, 2 = Jul–Sep, 3 = Oct–Dec, 4 = Jan–Mar  

- **Country_Avg_Processing**  
  Average processing time per nationality  

- **VisaType_Avg_Processing**  
  Average processing time per visa category  

- **Center_Load**  
  Number of applications processed by each processing center  

These features strengthen model learning and improve accuracy.

Final engineered dataset saved as:  
`visa_dataset_feature_engineered.csv`

## Technologies Used
- Python 3  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn  

## License
**MIT License**

## Author
**Swaathi B (GitHub: Swaathi1409)**


