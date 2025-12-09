import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import datetime
import io
import smtplib
from email.mime.text import MIMEText
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance


# ---------------- APP SETTINGS ----------------
st.set_page_config(page_title="Student Outcome Prediction", layout="wide")
DATA_PATH = "data 2.csv"   # your dataset file name


# ---------------- DATABASE ----------------
@st.cache_resource
def get_connection():
    con = sqlite3.connect("predictions.db", check_same_thread=False)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            predicted_label TEXT,
            model_used TEXT,
            inputs_json TEXT
        )
    """)
    con.commit()
    return con


# ---------------- LOAD & PREPROCESS DATA ----------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, sep=";")
    df = df.copy()
    df = df.dropna(subset=["Target"])
    df = df.reset_index(drop=True)
    return df


def get_features(df):
    return [c for c in df.columns if c not in ["Target", "id", "Unnamed: 0"]]


# ---------------- TRAIN MODELS ----------------
@st.cache_resource
def train_all_models(path):
    df = load_data(path)
    features = get_features(df)

    # Encode labels
    target_encoder = LabelEncoder()
    df["Target"] = target_encoder.fit_transform(df["Target"])
    y = df["Target"]

    # Encode categorical columns
    for col in features:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df[features].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    scale_models = {"Logistic Regression", "KNN", "SVM", "Neural Network"}

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=15),
        "Random Forest": RandomForestClassifier(n_estimators=220, max_depth=25),
        "KNN": KNeighborsClassifier(n_neighbors=12),
        "SVM": SVC(kernel="rbf", probability=True),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600),
    }

    results, trained = [], {}
    for name, model in models.items():
        if name in scale_models:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        trained[name] = model
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, preds, average="weighted", zero_division=0),
        })

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)

    # Feature importance for best model
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained[best_model_name]
    X_used = X_test_s if best_model_name in scale_models else X_test
    perm = permutation_importance(best_model, X_used, y_test, n_repeats=8, random_state=42)

    fi_df = pd.DataFrame({"Feature": features, "Importance": perm.importances_mean})
    fi_df = fi_df.sort_values("Importance", ascending=False)

    return {
        "df": df,
        "features": features,
        "scaler": scaler,
        "scale_models": scale_models,
        "models": trained,
        "results_df": results_df,
        "target_encoder": target_encoder,
        "feature_importance": fi_df
    }


# ---------------- MAIN APP ----------------
def main():
    st.title("🎓 Student Outcome Prediction Dashboard")

    data = train_all_models(DATA_PATH)
    df, features = data["df"], data["features"]

    page = st.sidebar.radio("📌 Navigation", [
        "Overview", "Model Performance", "Feature Importance", "Predict Outcome", "History"
    ])

    # ---------------- OVERVIEW ----------------
    if page == "Overview":
        st.subheader("📍 Dataset Preview")
        st.dataframe(df.head())
        st.write(f"📌 Total rows: **{len(df)}**")
        st.bar_chart(df["Target"].value_counts())

    # ---------------- MODEL PERFORMANCE ----------------
    elif page == "Model Performance":
        st.subheader("📌 ML Model Comparison")
        st.dataframe(data["results_df"].style.highlight_max(axis=0))
        st.success(f"🏆 Best Model: **{data['results_df'].iloc[0]['Model']}**")

    # ---------------- FEATURE IMPORTANCE ----------------
    elif page == "Feature Importance":
        st.subheader("📌 Global Feature Importance (Permutation Importance)")
        st.bar_chart(data["feature_importance"].set_index("Feature")["Importance"])
        st.dataframe(data["feature_importance"].reset_index(drop=True))

    # ---------------- PREDICT OUTCOME ----------------
    elif page == "Predict Outcome":
        st.subheader("🔮 Predict Student Outcome")

        inputs = {}
        for f in features:
            default = float(df[f].mean())
            inputs[f] = st.number_input(f, value=default)

        model_name = st.selectbox("Select Model", list(data["models"].keys()))
        model = data["models"][model_name]
        needs_scaling = model_name in data["scale_models"]

        if st.button("Predict"):
            X_new = pd.DataFrame([inputs], columns=features)
            if needs_scaling:
                X_new = data["scaler"].transform(X_new)

            pred = model.predict(X_new)[0]
            label = data["target_encoder"].inverse_transform([pred])[0]
            st.success(f"🎓 Predicted Result → **{label}**")

            # store in database
            con = get_connection()
            con.execute(
                "INSERT INTO predictions (timestamp, predicted_label, model_used, inputs_json) VALUES (?,?,?,?)",
                (str(datetime.datetime.now()), label, model_name, json.dumps(inputs)),
            )
            con.commit()

            # PDF export
            pdf = io.BytesIO()
            p = canvas.Canvas(pdf, pagesize=A4)
            p.drawString(100, 800, "Student Outcome Prediction Report")
            p.drawString(100, 780, f"Prediction: {label}")
            p.drawString(100, 760, f"Model: {model_name}")
            y = 730
            for k, v in inputs.items():
                p.drawString(100, y, f"{k}: {v}")
                y -= 15
            p.save()
            pdf.seek(0)

            st.download_button("⬇️ Download PDF", pdf, "prediction.pdf", mime="application/pdf")

            # Explain prediction (global importance)
            st.markdown("#### 🔍 Explanation (Feature Importance)")
            st.bar_chart(data["feature_importance"].set_index("Feature")["Importance"])

            # Email notification
            st.markdown("#### 📧 Send Email Notification")
            receiver = st.text_input("Recipient Email")
            if receiver and st.button("Send Email"):
                try:
                    sender = st.secrets["EMAIL_ADDRESS"]
                    password = st.secrets["EMAIL_PASSWORD"]
                    body = f"Prediction: {label}\nModel: {model_name}\n\nInput values:\n{json.dumps(inputs, indent=2)}"
                    msg = MIMEText(body)
                    msg["Subject"] = "Student Prediction Result"
                    msg["From"] = sender
                    msg["To"] = receiver

                    with smtplib.SMTP("smtp.gmail.com", 587) as s:
                        s.starttls()
                        s.login(sender, password)
                        s.sendmail(sender, receiver, msg.as_string())
                    st.success("📨 Email sent successfully!")
                except Exception as e:
                    st.error("Email sending failed.")
                    st.text(str(e))

    # ---------------- HISTORY ----------------
    elif page == "History":
        st.subheader("📁 Stored Prediction History")
        con = get_connection()
        df_history = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", con)
        if df_history.empty:
            st.info("No stored predictions yet.")
        else:
            st.dataframe(df_history)


if __name__ == "__main__":
    main()
