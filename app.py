
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Predictor")
st.write(
    "Explore how academic performance and experience "
    "relate to placement outcomes."
)

@st.cache_resource
def train_model():
    df = pd.read_csv("placement.csv")
    df.columns = df.columns.str.strip()

    if "status" not in df.columns:
        raise ValueError("Dataset must contain a 'status' column.")

    # Remove identifiers and post-placement information.
    df = df.drop(columns=["sl_no", "salary"], errors="ignore")
    df = df.dropna(subset=["status"])

    # Target labels in the standard dataset are Placed / Not Placed.
    X = df.drop(columns=["status"])
    y = df["status"].astype(str).str.strip()

    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical),

        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "F1-score": f1_score(
            y_test,
            predictions,
            pos_label="Placed"
        ) if "Placed" in y.unique() else f1_score(
            y_test, predictions, average="weighted"
        )
    }

    return model, X, metrics


try:
    model, X, metrics = train_model()
except Exception as e:
    st.error(f"Could not load or train the model: {e}")
    st.stop()

st.subheader("Model evaluation")
c1, c2 = st.columns(2)
c1.metric("Test accuracy", f"{metrics['Accuracy']:.1%}")
c2.metric("Test F1-score", f"{metrics['F1-score']:.3f}")

st.caption(
    "Metrics are calculated on a held-out test set. "
    "They are not a guarantee of future placement outcomes."
)

st.divider()
st.subheader("Predict for a student")
st.write("Enter values for the student's academic profile.")

categorical = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical = X.select_dtypes(
    include=["number"]
).columns.tolist()

with st.form("prediction_form"):
    values = {}
    columns = st.columns(2)

    for i, col in enumerate(X.columns):
        with columns[i % 2]:
            if col in categorical:
                options = sorted(
                    X[col].dropna().astype(str).unique().tolist()
                )
                if options:
                    values[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        options
                    )
            else:
                series = pd.to_numeric(X[col], errors="coerce")
                default = float(series.median())
                values[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=default
                )

    submitted = st.form_submit_button(
        "Predict placement",
        type="primary",
        use_container_width=True
    )

if submitted:
    student = pd.DataFrame(
        [{col: values[col] for col in X.columns}]
    )

    prediction = model.predict(student)[0]

    if prediction.lower() == "placed":
        st.success(f"Predicted outcome: {prediction}")
    else:
        st.warning(f"Predicted outcome: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(student)[0]
        classes = model.named_steps["classifier"].classes_

        st.write("Model probability estimates")
        for label, probability in zip(classes, probabilities):
            st.write(f"**{label}:** {probability:.1%}")
            st.progress(float(probability))
