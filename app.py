import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import datetime
import io
import smtplib
from email.mime.text import MIMEText

import shap
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


# ---------------- BASIC CONFIG ----------------
st.set_page_config(
    page_title="Student Outcome Prediction Dashboard",
    layout="wide"
)
DATA_PATH = "data 2.csv"


# ---------------- DATABASE CONNECTION ----------------
@st.cache_resource
def get_connection():
    con = sqlite3.connect("predictions.db", check_same_thread=False)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            predicted_label TEXT,
            model_used TEXT,
            inputs_json TEXT
        )
        """
    )
    con.commit()
    return con


# ---------------- DATA LOADING / FEATURES ----------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = df.copy()

    if "Age at enrollment" in df.columns:
        df = df[(df["Age at enrollment"] >= 15) & (df["Age at enrollment"] <= 60)]
    if "Previous qualification (grade)" in df.columns:
        df = df[
            (df["Previous qualification (grade)"] >= 0)
            & (df["Previous qualification (grade)"] <= 200)
        ]
    df = df.dropna(subset=["Target"])
    df = df.reset_index(drop=True)
    return df


def get_feature_list(df: pd.DataFrame):
    expanded = [
        "Previous qualification",
        "Previous qualification (grade)",
        "Application mode",
        "Application order",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Daytime/evening attendance",
        "Course",
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
        "Gender",
        "Nationality",
        "Marital status",
        "Age at enrollment",
        "Displaced",
        "Educational special needs",
        "Scholarship holder",
        "Debtor",
        "Tuition fees up to date",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]
    return [f for f in expanded if f in df.columns]


# ---------------- MODEL TRAINING ----------------
@st.cache_resource
def train_models(path: str):
    df = load_data(path)
    features = get_feature_list(df)

    # Encode categorical features
    for col in features:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["Target"])
    X = df[features].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scale_models = {"Logistic Regression", "KNN", "SVM", "Neural Network"}

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2500, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=14, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=25, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(96, 48),
            max_iter=600,
            random_state=42,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        if name in scale_models:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        trained_models[name] = model
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "Recall": recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
            }
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "df": df,
        "features": features,
        "results_df": results_df,
        "models": trained_models,
        "best_model": results_df.iloc[0]["Model"],
        "scaler": scaler,
        "target_encoder": target_encoder,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_scaled": X_test_scaled,
        "scale_models": scale_models,
    }


# ---------------- THEME / CSS ----------------
def apply_theme():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 38px;
            color: #0B5394;
            font-weight: 800;
            text-align: center;
            margin-bottom: 6px;
        }
        .sub-title {
            font-size: 23px;
            color: #0B5394;
            margin-top: 18px;
            margin-bottom: 10px;
            font-weight: 700;
        }
        .section-box {
            background: rgba(255,255,255,0.97);
            padding: 22px;
            border-radius: 14px;
            border-left: 6px solid #0B5394;
            box-shadow: 0 0 22px rgba(0,0,0,0.18);
            margin-bottom: 26px;
        }
        .metric-box {
            text-align: center;
            padding: 14px 8px;
            background: rgba(231,240,254,0.95);
            border-radius: 14px;
            border: 1px solid #9CC0FF;
            font-weight: 600;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------- MAIN APP ----------------
def main():
    apply_theme()

    # Train models & prepare data
    data = train_models(DATA_PATH)
    df = data["df"]
    features = data["features"]
    models = data["models"]
    results_df = data["results_df"]
    best_model_name = data["best_model"]
    scaler = data["scaler"]
    target_enc = data["target_encoder"]
    scale_models = data["scale_models"]

    # Title
    st.markdown(
        '<div class="main-title">🎓 Student Outcome Prediction Dashboard</div>',
        unsafe_allow_html=True,
    )

    # Highlight banner under title
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg,#0B5394,#1c7ed6);
            padding: 10px 14px;
            border-radius: 10px;
            margin: 6px 0 20px 0;
            text-align: center;
        ">
            <span style="color:white;font-size:19px;font-weight:600;">
                📊 AI-powered Dropout / Graduation Prediction &amp; Analytics  
                &nbsp;|&nbsp; Current Best Model: {best_model_name}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Model Performance",
            "Feature Importance",
            "Predict Outcome",
            "Prediction History",
        ],
    )

    # ---------- OVERVIEW ----------
    if page == "Overview":
        st.markdown(
            '<div class="sub-title">📍 Dataset Overview</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-box">', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f"<div class='metric-box'>Total Records<br><b>{len(df):,}</b></div>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<div class='metric-box'>Features Used<br><b>{len(features)}</b></div>",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"<div class='metric-box'>Target Classes<br><b>{df['Target'].nunique()}</b></div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Sample Records")
        st.dataframe(df.head())

        st.markdown("### Target Distribution")
        target_counts = df["Target"].value_counts().reset_index()
        target_counts.columns = ["Target", "Count"]
        st.bar_chart(target_counts.set_index("Target"))

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- MODEL PERFORMANCE ----------
    elif page == "Model Performance":
        st.markdown(
            '<div class="sub-title">📌 Model Comparison</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-box">', unsafe_allow_html=True)

        st.markdown("### Evaluation Metrics")
        st.dataframe(results_df.style.highlight_max(axis=0, color="#C6F6D5"))

        st.markdown("### Accuracy Comparison")
        st.bar_chart(results_df.set_index("Model")["Accuracy"])

        st.success(f"🏆 Best Performing Model: **{best_model_name}**")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- FEATURE IMPORTANCE ----------
    elif page == "Feature Importance":
        st.markdown(
            '<div class="sub-title">📌 Feature Importance</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-box">', unsafe_allow_html=True)

        best_model = models[best_model_name]
        X_test = data["X_test_scaled"] if best_model_name in scale_models else data["X_test"]
        y_test = data["y_test"]

        try:
            perm = permutation_importance(
                best_model, X_test, y_test, n_repeats=10, random_state=42
            )
            fi = pd.DataFrame(
                {"Feature": features, "Importance": perm.importances_mean}
            ).sort_values("Importance", ascending=False)

            st.markdown("### Global Feature Importance (Permutation Based)")
            st.bar_chart(fi.set_index("Feature")["Importance"])
            st.markdown("### Detailed Table")
            st.dataframe(fi.reset_index(drop=True))
        except Exception as e:
            st.error(f"Could not compute feature importance: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- PREDICT OUTCOME ----------
    elif page == "Predict Outcome":
        st.markdown(
            '<div class="sub-title">🔮 Predict Student Outcome</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-box">', unsafe_allow_html=True)

        model_name = st.selectbox(
            "Select model for prediction",
            list(models.keys()),
            index=list(models.keys()).index(best_model_name),
        )
        model = models[model_name]
        needs_scaling = model_name in scale_models

        st.markdown("#### Enter Student Information")
        inputs = {}
        with st.form("prediction_form"):
            for f in features:
                default_val = float(df[f].mean())
                inputs[f] = st.number_input(f, value=default_val)
            submitted = st.form_submit_button("Predict")

        if submitted:
            X_new = pd.DataFrame([inputs], columns=features)
            X_model = scaler.transform(X_new) if needs_scaling else X_new

            pred = model.predict(X_model)[0]
            label = target_enc.inverse_transform([pred])[0]

            st.success(f"🧾 Predicted Result → **{label}**")

            # Save prediction in DB
            con = get_connection()
            con.cursor().execute(
                "INSERT INTO predictions(timestamp,predicted_label,model_used,inputs_json) "
                "VALUES (?,?,?,?)",
                (
                    str(datetime.datetime.now()),
                    label,
                    model_name,
                    json.dumps(inputs),
                ),
            )
            con.commit()

            # ---- PDF report download ----
            st.markdown("#### 📄 Download Prediction Report (PDF)")
            pdf_buffer = io.BytesIO()
            pdf = canvas.Canvas(pdf_buffer, pagesize=A4)
            width, height = A4

            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(50, height - 50, "STUDENT OUTCOME PREDICTION REPORT")

            pdf.setFont("Helvetica", 12)
            pdf.drawString(50, height - 80, f"Predicted Result: {label}")
            pdf.drawString(50, height - 100, f"Model Used: {model_name}")
            pdf.drawString(50, height - 120, f"Timestamp: {datetime.datetime.now()}")

            y_pos = height - 150
            pdf.drawString(50, y_pos, "Input Features:")
            y_pos -= 20
            for k, v in inputs.items():
                pdf.drawString(60, y_pos, f"{k}: {v}")
                y_pos -= 15
                if y_pos < 40:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 12)
                    y_pos = height - 50

            pdf.save()
            pdf_buffer.seek(0)

            st.download_button(
                "⬇️ Download PDF Report",
                pdf_buffer,
                "prediction_report.pdf",
                mime="application/pdf",
            )

            # ---- Email notification ----
            st.markdown("#### 📧 Send Email Notification")
            receiver = st.text_input("Recipient email address")
            if receiver:
                if st.button("Send Email Notification"):
                    sender_email = "YOUR_GMAIL@gmail.com"
                    sender_app_password = "YOUR_APP_PASSWORD"  # Gmail app password

                    body = f"""Student Outcome Prediction

Predicted Result: {label}
Model Used: {model_name}
Timestamp: {datetime.datetime.now()}

Input Details:
{json.dumps(inputs, indent=2)}
"""
                    msg = MIMEText(body)
                    msg["Subject"] = "Student Outcome Prediction Report"
                    msg["From"] = sender_email
                    msg["To"] = receiver

                    try:
                        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
                        server.login(sender_email, sender_app_password)
                        server.sendmail(sender_email, receiver, msg.as_string())
                        server.quit()
                        st.success(f"📨 Email sent successfully to {receiver}")
                    except Exception as e:
                        st.error(f"Email sending failed: {e}")

            # ---- SHAP explainability ----
            with st.expander("🔍 Explain Prediction (SHAP – Random Forest)"):
                if model_name == "Random Forest":
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_new)
                        # For multi-class, use the predicted class index
                        shap_row = shap_values[pred][0]
                        shap_df = pd.DataFrame(
                            {
                                "Feature": features,
                                "Value": X_new.iloc[0].values,
                                "SHAP": shap_row,
                            }
                        )
                        shap_df["|SHAP|"] = np.abs(shap_df["SHAP"])
                        shap_df = shap_df.sort_values("|SHAP|", ascending=False)

                        st.markdown("Top contributing features")
                        st.dataframe(shap_df[["Feature", "Value", "SHAP", "|SHAP|"]])

                        st.markdown("Absolute SHAP impact")
                        st.bar_chart(shap_df.set_index("Feature")["|SHAP|"])
                    except Exception as e:
                        st.error(f"Could not compute SHAP values: {e}")
                else:
                    st.info(
                        "SHAP explanation is available only for the **Random Forest** model."
                    )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- PREDICTION HISTORY ----------
    elif page == "Prediction History":
        st.markdown(
            '<div class="sub-title">📁 Prediction History</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-box">', unsafe_allow_html=True)

        con = get_connection()
        df_hist = pd.read_sql(
            "SELECT * FROM predictions ORDER BY id DESC", con
        )
        if df_hist.empty:
            st.info("No predictions stored yet.")
        else:
            st.dataframe(df_hist)

        st.markdown("</div>", unsafe_allow_html=True)


# ------------ RUN APP ------------
if __name__ == "__main__":
    main()
