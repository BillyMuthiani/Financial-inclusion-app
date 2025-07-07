# Installing necessary packages
# Run in terminal:
# pip install pandas numpy scikit-learn xgboost pandas-profiling streamlit ydata-profiling

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
#from ydata_profiling import ProfileReport
import xgboost as xgb
import streamlit as st
import uuid

# Loading the dataset
data = pd.read_csv('Financial_inclusion_dataset.csv')

# Displaying general information about the dataset
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())
print("\nSummary Statistics:")
print(data.describe(include='all'))

# Generating pandas profiling report
#profile = ProfileReport(data, title="Financial Inclusion Profiling Report", explorative=True)
#profile.to_file("financial_inclusion_report.html")

# Handling missing and corrupted values
print("\nMissing Values:")
print(data.isnull().sum())
# No missing values observed in the dataset based on provided sample

# Removing duplicates
data = data.drop_duplicates()
print(f"\nShape after removing duplicates: {data.shape}")

# Handling outliers for numerical columns (household_size, age_of_respondent)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

data = remove_outliers(data, 'household_size')
data = remove_outliers(data, 'age_of_respondent')
print(f"\nShape after handling outliers: {data.shape}")

# Encoding categorical features
le = LabelEncoder()
categorical_cols = ['country', 'location_type', 'cellphone_access', 'gender_of_respondent',
                    'relationship_with_head', 'marital_status', 'education_level', 'job_type']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Encoding target variable
data['bank_account'] = le.fit_transform(data['bank_account'])

# Preparing features and target
X = data.drop(['bank_account', 'uniqueid', 'year'], axis=1)
y = data['bank_account']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training XGBoost with hyperparameter tuning
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
param_grid = {
    'n_estimators': [100],
    'max_depth': [3],
    'learning_rate': [ 0.1],
    'subsample': [0.8]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# Evaluating on test set
y_pred = best_model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Saving the model
best_model.save_model('xgboost_financial_inclusion.json')

# Streamlit application
st.title("Financial Inclusion Prediction App")

st.write("Enter the details below to predict bank account ownership:")

# Creating input fields for features
country = st.selectbox("Country", ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
location_type = st.selectbox("Location Type", ['Rural', 'Urban'])
cellphone_access = st.selectbox("Cellphone Access", ['Yes', 'No'])
household_size = st.number_input("Household Size", min_value=1, max_value=20, value=3)
age_of_respondent = st.number_input("Age of Respondent", min_value=16, max_value=100, value=30)
gender_of_respondent = st.selectbox("Gender", ['Male', 'Female'])
relationship_with_head = st.selectbox("Relationship with Head", ['Head of Household', 'Spouse', 'Child', 'Parent', 'Other relative', 'Other non-relatives'])
marital_status = st.selectbox("Marital Status", ['Married/Living together', 'Single/Never Married', 'Widowed', 'Divorced/Seperated', 'Dont know'])
education_level = st.selectbox("Education Level", ['No formal education', 'Primary education', 'Secondary education', 'Vocational/Specialised training', 'Tertiary education', 'Other/Dont know/RTA'])
job_type = st.selectbox("Job Type", ['Self employed', 'Informally employed', 'Farming and Fishing', 'Remittance Dependent', 'Other Income', 'Formally Employed Private', 'Formally Employed Government', 'Government Dependent', 'No Income', 'Dont Know/Refuse to answer'])

# Button for prediction
if st.button("Predict"):
    # Encoding input data
    input_data = pd.DataFrame({
        'country': [country],
        'location_type': [location_type],
        'cellphone_access': [cellphone_access],
        'household_size': [household_size],
        'age_of_respondent': [age_of_respondent],
        'gender_of_respondent': [gender_of_respondent],
        'relationship_with_head': [relationship_with_head],
        'marital_status': [marital_status],
        'education_level': [education_level],
        'job_type': [job_type]
    })

    # Encoding categorical inputs using the same LabelEncoder
    for col in categorical_cols:
        input_data[col] = le.fit_transform(input_data[col])

    # Making prediction
    prediction = best_model.predict(input_data)
    result = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f"Prediction: The individual is likely to have a bank account: {result}")

# Deployment Instructions
