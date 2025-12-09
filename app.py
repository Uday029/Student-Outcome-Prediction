{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import numpy as np\
\
from sklearn.model_selection import train_test_split\
from sklearn.preprocessing import StandardScaler, LabelEncoder\
from sklearn.linear_model import LogisticRegression\
from sklearn.tree import DecisionTreeClassifier\
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier\
from sklearn.neighbors import KNeighborsClassifier\
from sklearn.svm import SVC\
from sklearn.neural_network import MLPClassifier\
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\
\
# ----------------- CONFIG -----------------\
DATA_PATH = "data 2.csv"\
\
st.set_page_config(\
    page_title="Student Outcome Prediction Dashboard",\
    layout="wide"\
)\
\
\
# ----------------- DATA LOADING -----------------\
@st.cache_data\
def load_data(path: str) -> pd.DataFrame:\
    df = pd.read_csv(path, sep=";")\
    df = df.copy()\
\
    # Basic cleaning similar to notebook\
    if "Age at enrollment" in df.columns:\
        df = df[(df["Age at enrollment"] >= 15) &\
                (df["Age at enrollment"] <= 60)]\
\
    if "Previous qualification (grade)" in df.columns:\
        df = df[(df["Previous qualification (grade)"] >= 0) &\
                (df["Previous qualification (grade)"] <= 200)]\
\
    if "Admission grade" in df.columns:\
        df = df[(df["Admission grade"] >= 0) &\
                (df["Admission grade"] <= 200)]\
\
    # Remove rows with missing Target\
    df = df.dropna(subset=["Target"])\
    df = df.reset_index(drop=True)\
    return df\
\
\
def get_feature_list(df: pd.DataFrame):\
    """\
    Feature list inspired by your notebook:\
    Age, admission grades, semester performance, scholarship etc.\
    """\
    base_features = [\
        "Age at enrollment",\
        "Admission grade",\
        "Curricular units 1st sem (credited)",\
        "Curricular units 1st sem (enrolled)",\
        "Curricular units 1st sem (evaluations)",\
        "Curricular units 1st sem (approved)",\
        "Curricular units 1st sem (grade)",\
        "Curricular units 1st sem (without evaluations)",\
        "Curricular units 2nd sem (credited)",\
        "Curricular units 2nd sem (enrolled)",\
        "Curricular units 2nd sem (evaluations)",\
        "Curricular units 2nd sem (approved)",\
        "Curricular units 2nd sem (grade)",\
        "Curricular units 2nd sem (without evaluations)",\
        "Scholarship holder",\
        "Tuition fees up to date",\
        "Debtor",\
        "Displaced",\
        "Educational special needs",\
        "Unemployment rate",\
        "Inflation rate",\
        "GDP",\
    ]\
    # Keep only features that exist in the dataset\
    return [f for f in base_features if f in df.columns]\
\
\
# ----------------- TRAINING PIPELINE -----------------\
@st.cache_resource\
def prepare_and_train(path: str):\
    df = load_data(path)\
    features = get_feature_list(df)\
\
    if len(features) == 0:\
        raise ValueError("No matching feature columns found in dataset.")\
\
    # Encode Target (Graduate / Dropout / Enrolled)\
    label_encoder = LabelEncoder()\
    y = label_encoder.fit_transform(df["Target"])\
    X = df[features].astype(float)\
\
    X_train, X_test, y_train, y_test = train_test_split(\
        X, y, test_size=0.2, random_state=42, stratify=y\
    )\
\
    scaler = StandardScaler()\
    X_train_scaled = scaler.fit_transform(X_train)\
    X_test_scaled = scaler.transform(X_test)\
\
    # Models that require scaling\
    scale_models = \{\
        "Logistic Regression",\
        "K-Nearest Neighbors",\
        "Support Vector Machine",\
        "Neural Network (MLP)",\
    \}\
\
    # Same family of models as your notebook\
    models = \{\
        "Logistic Regression": LogisticRegression(\
            random_state=42, max_iter=2000, multi_class="ovr"\
        ),\
        "Decision Tree": DecisionTreeClassifier(\
            random_state=42, max_depth=10\
        ),\
        "Random Forest": RandomForestClassifier(\
            n_estimators=150, random_state=42, max_depth=20\
        ),\
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),\
        "Support Vector Machine": SVC(\
            kernel="rbf", probability=True, random_state=42\
        ),\
        "Neural Network (MLP)": MLPClassifier(\
            hidden_layer_sizes=(64, 32),\
            activation="relu",\
            solver="adam",\
            max_iter=500,\
            random_state=42,\
            early_stopping=True,\
        ),\
    \}\
\
    results = []\
    fitted_models = \{\}\
\
    for name, model in models.items():\
        if name in scale_models:\
            model.fit(X_train_scaled, y_train)\
            y_pred = model.predict(X_test_scaled)\
        else:\
            model.fit(X_train, y_train)\
            y_pred = model.predict(X_test)\
\
        acc = accuracy_score(y_test, y_pred)\
        prec = precision_score(\
            y_test, y_pred, average="weighted", zero_division=0\
        )\
        rec = recall_score(\
            y_test, y_pred, average="weighted", zero_division=0\
        )\
        f1 = f1_score(\
            y_test, y_pred, average="weighted", zero_division=0\
        )\
\
        results.append(\
            \{\
                "Model": name,\
                "Accuracy": acc,\
                "Precision": prec,\
                "Recall": rec,\
                "F1-Score": f1,\
            \}\
        )\
        fitted_models[name] = model\
\
    results_df = pd.DataFrame(results).sort_values(\
        "Accuracy", ascending=False\
    ).reset_index(drop=True)\
    best_model_name = results_df.iloc[0]["Model"]\
\
    return \{\
        "df": df,\
        "features": features,\
        "X_train": X_train,\
        "X_test": X_test,\
        "y_train": y_train,\
        "y_test": y_test,\
        "X_train_scaled": X_train_scaled,\
        "X_test_scaled": X_test_scaled,\
        "scaler": scaler,\
        "label_encoder": label_encoder,\
        "results_df": results_df,\
        "models": fitted_models,\
        "scale_models": scale_models,\
        "best_model_name": best_model_name,\
    \}\
\
\
# ----------------- STREAMLIT APP -----------------\
def main():\
    st.title("\uc0\u55356 \u57235  Student Outcome Prediction Dashboard")\
    st.write(\
        "This app uses the same machine-learning pipeline as your Jupyter "\
        "notebook to analyse student data and predict outcomes "\
        "like **Graduate**, **Dropout**, or **Enrolled**."\
    )\
\
    # Train / load models\
    try:\
        data = prepare_and_train(DATA_PATH)\
    except Exception as e:\
        st.error(f"Error while loading data or training models: \{e\}")\
        return\
\
    df = data["df"]\
    features = data["features"]\
    results_df = data["results_df"]\
    models = data["models"]\
    scale_models = data["scale_models"]\
    scaler = data["scaler"]\
    label_encoder = data["label_encoder"]\
\
    page = st.sidebar.radio(\
        "\uc0\u55357 \u56524  Navigation",\
        ["Overview", "Model Performance", "Predict Outcome"],\
    )\
\
    # ---------- OVERVIEW PAGE ----------\
    if page == "Overview":\
        st.subheader("Dataset Overview")\
\
        col1, col2, col3, col4 = st.columns(4)\
        with col1:\
            st.metric("Total records", f"\{len(df):,\}")\
        with col2:\
            st.metric("Number of features", str(len(features)))\
        with col3:\
            st.metric("Target classes", str(df["Target"].nunique()))\
        with col4:\
            class_counts = df["Target"].value_counts()\
            majority_class = class_counts.idxmax()\
            st.metric("Most common outcome", majority_class)\
\
        st.markdown("### Sample Data")\
        st.dataframe(df.head())\
\
        st.markdown("### Target Distribution")\
        target_counts = df["Target"].value_counts().reset_index()\
        target_counts.columns = ["Target", "Count"]\
        st.bar_chart(target_counts.set_index("Target"))\
\
    # ---------- MODEL PERFORMANCE PAGE ----------\
    elif page == "Model Performance":\
        st.subheader("Model Comparison")\
\
        st.markdown(\
            "The following models were trained using the same preprocessing "\
            "pipeline as in your notebook:"\
        )\
        st.dataframe(\
            results_df.style.highlight_max(axis=0, color="#c6f6d5")\
        )\
\
        st.markdown("### Accuracy Comparison")\
        st.bar_chart(results_df.set_index("Model")["Accuracy"])\
\
        st.markdown("### Detailed Metrics (Precision / Recall / F1)")\
        st.bar_chart(\
            results_df.set_index("Model")[["Precision", "Recall", "F1-Score"]]\
        )\
\
        st.success(\
            f"Best model based on accuracy: **\{data['best_model_name']\}**"\
        )\
\
    # ---------- PREDICTION PAGE ----------\
    elif page == "Predict Outcome":\
        st.subheader("\uc0\u55357 \u56622  Predict Student Outcome")\
\
        st.markdown(\
            "Enter student information below. The app will use the trained "\
            "model to predict whether the student is likely to **Graduate**, "\
            "**Dropout**, or remain **Enrolled**."\
        )\
\
        selected_model_name = st.selectbox(\
            "Choose model for prediction",\
            options=list(models.keys()),\
            index=list(models.keys()).index(data["best_model_name"]),\
        )\
        model = models[selected_model_name]\
        needs_scaling = selected_model_name in scale_models\
\
        with st.form("prediction_form"):\
            input_values = \{\}\
\
            for feature in features:\
                col_min = float(df[feature].min())\
                col_max = float(df[feature].max())\
                col_mean = float(df[feature].mean())\
                step = 1.0\
\
                # If column has decimals, use smaller step\
                if not np.allclose(df[feature], df[feature].round()):\
                    step = 0.1\
\
                # Binary fields as Yes/No\
                if feature in [\
                    "Scholarship holder",\
                    "Tuition fees up to date",\
                    "Debtor",\
                    "Displaced",\
                    "Educational special needs",\
                ]:\
                    default_val = int(round(col_mean))\
                    label = feature.replace("_", " ")\
                    choice = st.selectbox(\
                        label,\
                        options=["No", "Yes"],\
                        index=1 if default_val == 1 else 0,\
                        help=f"\{feature\} (0 = No, 1 = Yes)",\
                    )\
                    input_values[feature] = 1 if choice == "Yes" else 0\
                else:\
                    input_values[feature] = st.number_input(\
                        feature,\
                        min_value=col_min,\
                        max_value=col_max,\
                        value=col_mean,\
                        step=step,\
                    )\
\
            submitted = st.form_submit_button("Predict")\
\
        if submitted:\
            X_new = pd.DataFrame([input_values], columns=features)\
\
            if needs_scaling:\
                X_new_transformed = scaler.transform(X_new)\
            else:\
                X_new_transformed = X_new\
\
            pred_encoded = model.predict(X_new_transformed)[0]\
\
            # Try to show probabilities if the model supports it\
            try:\
                proba = model.predict_proba(X_new_transformed)[0]\
                class_labels = label_encoder.inverse_transform(\
                    np.arange(len(proba))\
                )\
                proba_df = pd.DataFrame(\
                    \{"Outcome": class_labels, "Probability": proba\}\
                ).sort_values("Probability", ascending=False)\
            except Exception:\
                proba_df = None\
\
            predicted_label = label_encoder.inverse_transform(\
                [pred_encoded]\
            )[0]\
\
            st.markdown("### Prediction Result")\
            st.success(\
                f"\uc0\u55358 \u56830  Predicted outcome: **\{predicted_label\}** "\
                f"using **\{selected_model_name\}**"\
            )\
\
            if proba_df is not None:\
                st.markdown("### Class Probabilities")\
                st.dataframe(proba_df.reset_index(drop=True))\
\
            st.markdown("### Input Summary")\
            st.json(input_values)\
\
\
if __name__ == "__main__":\
    main()\
}