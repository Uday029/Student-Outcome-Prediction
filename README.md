# Student-Outcome-Prediction
machine-learning data-science streamlit education dropout-prediction student-performance shap university-dataset feature-importance
🧠 Student Outcome Prediction Project

End-to-End Machine Learning + Streamlit Dashboard

🔗 Live App: https://your-username-student-outcome.streamlit.app  


📌 Project Overview

This project predicts whether a student will **Graduate**, **Drop Out**, or **Continue Enrollment** using demographic, academic and socio-economic features such as age at enrollment, previous qualification grade, scholarship status, curricular performance, and more.

The goal is to demonstrate a full end-to-end ML workflow, including:

- Data cleaning & preprocessing  
- Exploratory data analysis  
- Feature engineering  
- Model building & comparison  
- Outlier handling  
- Model explainability (SHAP)  
- Streamlit web deployment  

---

📊 Dataset

Source: **University-provided higher education student dataset** (internal, not from Kaggle/UCI)  
Rows: *N* (fill after analysis, e.g. 4,500)  
Target Column: `Target` (Graduate / Dropout / Enrolled)

---

🧹 Data Preprocessing Steps

✔ Removed missing or incorrect entries  
✔ Filtered invalid ranges (e.g. age, previous qualification grade, admission grade)  
✔ Handled categorical variables using Label Encoding / One-Hot Encoding  
✔ Treated outliers in numerical columns  
✔ Scaled numerical features where needed (for models like Logistic Regression, SVM, KNN, MLP)  
✔ Train-test split (typically 80–20)

---

🔍 Exploratory Data Analysis

The notebook includes analysis and charts for:

- Age distribution of students  
- Distribution of Target classes (Graduate / Dropout / Enrolled)  
- Relationship between previous qualification grade and outcome  
- Curricular units (1st and 2nd semester) vs dropout risk  
- Socio-economic indicators (unemployment rate, inflation, GDP) vs student outcome  
- Correlation matrix of key numerical features  

Visualizations are saved inside:  
📁 `visualizations/`

---

🤖 Machine Learning Models

Models trained:

- Logistic Regression (with scaling)  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- MLP Neural Network  

Random Forest generally performed best and is used as the **primary model** for predictions in the Streamlit app.

(You can add a small table here later with your exact accuracies if you want.)

---

🎯 Prediction Logic

The prediction is driven by the trained ML model and uses features such as:

- Age at enrollment  
- Previous qualification & grades  
- Semester-wise curricular performance  
- Scholarship status / debtor / tuition fees up to date  
- Socio-economic context (unemployment rate, inflation, GDP)  

These inputs are passed through the best-performing model (e.g. Random Forest) to predict whether the student is more likely to Graduate, Drop Out, or remain Enrolled.

---

🌐 Streamlit Dashboard

The app includes:

- Input panel for entering student details  
- Live prediction of student outcome  
- Model results summary  
- Global feature importance (permutation importance)  
- SHAP-based explainability for individual predictions  
- PDF report download for each prediction  
- Email notification option  
- Prediction history stored using SQLite  

Main dashboard file:  
📄 `app/app.py`

---

📁 Project Structure

```text
Student-Outcome-Prediction/
│
├── notebooks/
│   └── Student Dropout & Academic Success Prediction.ipynb                # Main ML notebook (EDA + modeling)
│
├── app/
│   └── app.py                           # Streamlit dashboard
│
├── data/
│   └── data 2.csv           # University-provided dataset
│
├── visualizations/                      # Plots exported from notebook
│   └── (PNG images)
│
├── requirements.txt
├── .gitignore
└── README.md

