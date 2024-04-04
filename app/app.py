import streamlit as st
import pandas as pd
from app.plots import bar_chart, confusion_matrix
from io import StringIO
import joblib
from sklearn.metrics import recall_score, f1_score, accuracy_score


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    df_processed = pd.read_pickle("./data/df_processed.pkl")
    df = pd.read_pickle("./data/df.pkl")
    return {"df": df, "df_processed": df_processed}


@st.cache_resource
def load_models() -> dict[str, pd.DataFrame]:
    pipeline = joblib.load("./pipeline/feature_engineering_pipeline.pkl")
    model = joblib.load("./models/votation_classifier.pkl")
    return {"model": model, "pipeline": pipeline}


# @st.cache_data()
def running_model(df, pipeline, model, threshold=0.5):
    df_ = df.copy()
    churn_in = False
    if "Churn" in df.columns:
        churn_in = True
    else:
        churn_test = [
            ("Yes" if i < df.shape[0] / 2 else "No") for i in range(df.shape[0])
        ]
        df_["Churn"] = churn_test

    df_ = pipeline.transform(df_).copy()
    X = df_.drop("Churn", axis=1)
    # y = df_["Churn"]
    y_predict = model.predict_proba(X)[:, 1]
    df["Churn_prediction_proba"] = y_predict
    df["Churn_prediction"] = y_predict >= threshold
    df["Churn_number"] = df_["Churn"]

    if churn_in:
        accuracy = accuracy_score(df_["Churn"].astype(int), y_predict >= threshold)
        f1 = f1_score(df_["Churn"].astype(int), y_predict >= threshold)
        recall = recall_score(df_["Churn"].astype(int), y_predict >= threshold)
    else:
        accuracy = "N/A"
        f1 = "N/A"
        recall = "N/A"

    return {
        "df": df,
        "df_processed": df_,
        "churn_in": churn_in,
        "accuracy_score": accuracy,
        "f1_score": f1,
        "recall_score": recall,
    }


def main():
    st.set_page_config(
        page_title="Churn on Telecom",
        page_icon="🔵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    loaded = False
    with st.sidebar:
        st.write("# Parameters")
        # validator_data = st.radio("Data contains 'Churn' variable", ["No", "Yes"])
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
        input_data = st.radio("Select data", ["Upload Data", "Training Data"])
        if input_data == "Upload Data":
            uploaded_file = st.file_uploader("Choose a file", type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                loaded = True

    ## Load data
    if not loaded:
        data = load_data()
        df = data["df"]
        df_processed = data["df_processed"]
    ## Loading models
    models = load_models()
    model = models["model"]
    pipeline = models["pipeline"]

    ## Running model
    data_running_model = running_model(df, pipeline, model, threshold)
    df = data_running_model["df"]
    df_processed = data_running_model["df_processed"]
    churn_in = data_running_model["churn_in"]
    ##

    st.write("# Churn on Telecom")
    ## Indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Total quantity churn prediction",
        f'{df[df["Churn_prediction"]==1].shape[0]} ({df[df["Churn_prediction"]==1].shape[0]/df.shape[0]*100:.1f}%)',
    )
    col2.metric(
        "Total monthly charges by churn prediction",
        f'${df_processed["MonthlyCharges"].sum():,.1f}',
    )
    col3.metric(
        "Total charges by churn prediction",
        f'${df_processed["TotalCharges"].sum():,.1f}',
    )
    if churn_in:
        col4.metric("f1 score", f'{data_running_model["f1_score"]*100:.1f}%')
        # col5.metric("accuracy score", f'{data_running_model["accuracy_score"]*100:.1f}%')
        col5.metric("recall score", f'{data_running_model["recall_score"]*100:.1f}%')
    else:
        col4.metric("f1 score", f'{data_running_model["f1_score"]}')
        # col5.metric("accuracy score", f'{data_running_model["accuracy_score"]}')
        col5.metric("recall score", f'{data_running_model["recall_score"]}')

    if churn_in:
        labels = df["Churn"].value_counts().index
        sizes = df["Churn"].value_counts().values
    else:
        labels = df["Churn_prediction"].value_counts().index
        sizes = df["Churn_prediction"].value_counts().values

    st.divider()
    col1, col2 = st.columns(2)
    fig = bar_chart(labels, sizes)

    col1.write("##### Churn distribution")
    col1.plotly_chart(fig)
    data_export = df[
        ["customerID", "gender", "Churn_prediction", "Churn_prediction_proba"]
    ].sort_values("Churn_prediction_proba", ascending=False)
    col2.write("##### Confusion matrix")
    col2.pyplot(confusion_matrix(df["Churn_number"], df["Churn_prediction"]))

    st.divider()
    st.write("##### Churn dataset")
    st.dataframe(data_export[data_export["Churn_prediction"]], use_container_width=True)
