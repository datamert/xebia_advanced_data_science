import streamlit as st
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Load all referenced variables from disk (from 'joblib' subfolder)
df = joblib.load("joblib/df.joblib")
feature_importances_df = joblib.load("joblib/feature_importances_df.joblib")
y_test = joblib.load("joblib/y_test.joblib")
y_test_pred_reduced = joblib.load("joblib/y_test_pred_reduced.joblib")

st.title("San Francisco Crime Classification Project")

st.markdown("""
**Author:** Mert Gunes  
**Hackathon:** Xebia Advanced Data Science with Python (May 2025)  
---
### Project Summary
This project explores and models San Francisco crime data (2010-2015) to predict whether a crime is a theft.  
Key steps:
- Data cleaning and exploration
- Feature engineering (including custom transformers)
- Model building with pipelines and hyperparameter tuning
- Feature importance analysis and model simplification

### Data Overview
- **Target:** `theft` (binary)
- **Features:** Date/time, location, police district, address patterns, engineered features (e.g. mean thefts on main streets, block indicator, coordinate combinations, past theft resolution stats)
- **Rows:** {0}
- **Columns:** {1}
""".format(df.shape[0], df.shape[1]))

st.subheader("Feature Engineering Highlights")
st.markdown("""
- **Date features:** Extracted year, month, day, weekday, hour, minute, days since start
- **Address features:** Block indicator, mean thefts for main street (target encoding)
- **Location features:** x/y combinations
- **Past theft resolution:** Number of caught/not caught thefts in previous quarter
""")

st.subheader("Modeling & Results")
st.markdown("""
- **Model:** Random Forest (with pipeline and hyperparameter tuning)
- **Best accuracy:** {:.2f} (test set, top 5 features)
- **Most important feature:** `main_street__mean_thefts` (~33% importance)
- **Model simplification:** Top 5 features retained most of the predictive power
""".format(accuracy_score(y_test, y_test_pred_reduced)))

st.subheader("Feature Importances (Top 5)")
st.bar_chart(feature_importances_df.set_index('feature').head(5)['importance'])

st.subheader("Classification Report (Test Set)")
st.text(classification_report(y_test, y_test_pred_reduced))
