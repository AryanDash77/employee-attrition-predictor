import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap

with open('models/best_model_xgb.pkl','rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
threshold = model_package['threshold']
feature_names = model_package['feature_names']

with open('models/scaler.pkl','rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title = "Employee Attrition Predictor")
st.title("Employee Attrition Predictor")
st.write("Predict employee attrition risk and understand the key drivers behind each prediction.")


st.header("Employee Details")

col1 , col2 , col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
    distance_from_home = st.slider("Distance from Home (km)", 1, 30, 5)
    total_working_years = st.slider("Total Working Years", 0, 40, 5)
    years_at_company = st.slider("Years at Company", 0, 40, 3)

with col2:
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education_field = st.selectbox("Education Field", [
        "Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"
    ])
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

with col3:
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    stock_option_level = st.slider("Stock Option Level (0-3)", 0, 3, 1)


st.header("Prediction")

if st.button("Predict Attrition Risk"):
     # Step A: Start with default values for all 43 features
    input_dict = {
        'Age': age,
        'BusinessTravel': {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}[business_travel],
        'DailyRate': 800,  # dataset median-ish default
        'DistanceFromHome': distance_from_home,
        'Education': 3,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': 1,  # default Male, not collected in form
        'HourlyRate': 65,
        'JobInvolvement': 3,
        'JobLevel': 2,
        'JobSatisfaction': job_satisfaction,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': 14000,
        'NumCompaniesWorked': 2,
        'OverTime': 1 if overtime == "Yes" else 0,
        'PercentSalaryHike': 15,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': 2,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': min(years_at_company, 3),
        'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': min(years_at_company, 3),
    }

    # Step B: Add OneHot columns, all defaulted to 0
    onehot_cols = [f for f in feature_names if f.startswith(('Department_', 'EducationField_', 'JobRole_', 'MaritalStatus_'))]
    for col in onehot_cols:
        input_dict[col] = 0

    # Step C: Set the correct OneHot flags to 1 based on form selections
    dept_col = f'Department_{department}'
    if dept_col in input_dict:
        input_dict[dept_col] = 1

    role_col = f'JobRole_{job_role}'
    if role_col in input_dict:
        input_dict[role_col] = 1

    marital_col = f'MaritalStatus_{marital_status}'
    if marital_col in input_dict:
        input_dict[marital_col] = 1

    edu_field_col = f'EducationField_{education_field}'
    if edu_field_col in input_dict:
        input_dict[edu_field_col] = 1

    # Step D: Build dataframe in the EXACT column order the model expects
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]  # enforce correct order

    # Step E: Scale using the saved scaler
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

    # Step F: Predict using the saved threshold, not default 0.5
    probability = model.predict_proba(input_scaled_df)[:, 1][0]
    prediction = 1 if probability >= threshold else 0

    # Step G: Display result
    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ High Attrition Risk — Predicted probability: {probability:.1%}")
    else:
        st.success(f"✅ Low Attrition Risk — Predicted probability: {probability:.1%}")

