import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import smtplib
from email.mime.text import MIMEText
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os
from datetime import datetime

st.set_page_config(page_title="Student Outcome Prediction", layout="wide")

# --------------------- Load Model ---------------------
MODEL_PATH = "models/model.pkl"  # change if different
model = joblib.load(MODEL_PATH)

FEATURES = ['Age', 'Previous_Qualification_Grade', 'Admission_Grade',
            'Curricular_Units_1st_Sem_Appr', 'Curricular_Units_1st_Sem_Grade',
            'Curricular_Units_2nd_Sem_Appr', 'Curricular_Units_2nd_Sem_Grade',
            'Scholarship_Holder', 'Unemployment_Rate', 'Inflation_Rate', 'GDP']

# --------------------- Database Setup ---------------------
DB_FILE = "predictions.db"
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    timestamp TEXT,
    student_name TEXT,
    email TEXT,
    prediction TEXT
)
""")
conn.commit()

# --------------------- Input UI ---------------------
st.title("🎓 Student Outcome Prediction Dashboard")

col1, col2, col3 = st.columns(3)
student_name = col1.text_input("Student Name")
email_address = col2.text_input("Email Address")
age = col3.number_input("Age", 16, 50)

prev_grade = st.number_input("Previous Qualification Grade", 0, 200)
admission_grade = st.number_input("Admission Grade", 0, 200)
sem1_appr = st.number_input("Curricular Units 1st Sem Approved", 0, 20)
sem1_grade = st.number_input("Curricular Units 1st Sem Grade", 0, 20)
sem2_appr = st.number_input("Curricular Units 2nd Sem Approved", 0, 20)
sem2_grade = st.number_input("Curricular Units 2nd Sem Grade", 0, 20)
scholar = st.selectbox("Scholarship Holder", [0, 1])
unemployment = st.number_input("Unemployment Rate", 0.0, 50.0)
inflation = st.number_input("Inflation Rate", 0.0, 50.0)
gdp = st.number_input("GDP (Billion USD)", 0.0, 50000.0)

# --------------------- Prediction ---------------------
if st.button("🔮 Predict Outcome"):
    input_df = pd.DataFrame([[
        age, prev_grade, admission_grade,
        sem1_appr, sem1_grade,
        sem2_appr, sem2_grade,
        scholar, unemployment, inflation, gdp
    ]], columns=FEATURES)

    pred = model.predict(input_df)[0]
    outcome = {0: "Graduate", 1: "Dropout", 2: "Enrolled"}.get(pred, "Unknown")

    st.success(f"🎓 Prediction: {outcome}")

    # Save to DB
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO history VALUES (?, ?, ?, ?)",
                   (timestamp, student_name, email_address, outcome))
    conn.commit()

    st.write("---")
    st.subheader("📌 Prediction Explanation (Feature Importance)")

    try:
        # Compute permutation importance
        result = permutation_importance(
            model,
            input_df,
            model.predict(input_df),
            n_repeats=5,
            random_state=42
        )
        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": result.importances_mean
        }).sort_values("Importance", ascending=False)

        st.bar_chart(importance.set_index("Feature")["Importance"])

    except Exception as e:
        st.warning("Explanation unavailable for this prediction.")
        st.text(str(e))

# --------------------- Prediction History ---------------------
st.write("---")
st.subheader("📅 Prediction History")

cursor.execute("SELECT * FROM history")
rows = cursor.fetchall()
history_df = pd.DataFrame(rows, columns=["Time", "Name", "Email", "Prediction"])
st.dataframe(history_df)

# --------------------- PDF Generator ---------------------
def generate_pdf(name, prediction):
    filename = f"{name}_prediction.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    c.drawString(100, 780, "Student Outcome Prediction Report")
    c.drawString(100, 750, f"Student Name: {name}")
    c.drawString(100, 720, f"Prediction: {prediction}")
    c.save()
    return filename

if st.button("📄 Download Prediction PDF"):
    if student_name:
        pdf_file = generate_pdf(student_name, outcome)
        with open(pdf_file, "rb") as f:
            st.download_button("Download PDF", f, file_name=pdf_file)
    else:
        st.warning("Enter student name first.")

# --------------------- Email Notification ---------------------
if st.button("📧 Send Email Notification"):
    try:
        sender_email = st.secrets["EMAIL_ADDRESS"]
        sender_pass = st.secrets["EMAIL_PASSWORD"]
        msg = MIMEText(f"Student Name: {student_name}\nPrediction: {outcome}")
        msg["Subject"] = "Student Outcome Prediction Result"
        msg["From"] = sender_email
        msg["To"] = email_address

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, email_address, msg.as_string())

        st.success("📨 Email sent successfully!")
    except Exception as e:
        st.error("Email failed. Add EMAIL_ADDRESS + EMAIL_PASSWORD in Secrets.")
        st.text(str(e))
