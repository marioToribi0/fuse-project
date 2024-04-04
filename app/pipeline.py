from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


class ColumnDropperTransformer:
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self


class ConvertDataTransformer:
    def __init__(self) -> None:
        self.objects = [
            "customerID",
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "Churn",
        ]
        self.floats = ["MonthlyCharges", "TotalCharges"]
        self.ints = ["SeniorCitizen", "tenure"]

    def transform(self, X: pd.DataFrame, y=None):
        X_ = X.copy()
        X_[self.objects] = X_[self.objects].astype(str)
        for col_name in self.floats:
            X_[col_name] = pd.to_numeric(X_[col_name], errors="coerce")
        for col_name in self.ints:
            X_[col_name] = pd.to_numeric(X_[col_name], errors="coerce")
        return X_

    def fit(self, X, y=None):
        return self


class RenameColumns:
    def __init__(self) -> None:
        pass

    def transform(self, X: pd.DataFrame, y=None):
        X_ = X.copy()
        X_.rename(columns={"encode__Churn_Yes": "Churn"}, inplace=True)
        X_.drop("encode__Churn_No", axis=1, inplace=True)
        X_.columns = [
            col_name.replace("encode__", "").replace("remainder__", "")
            for col_name in X_.columns
        ]
        X_.fillna(0, inplace=True)
        return X_

    def fit(self, X, y=None):
        return self


object_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Churn",
]
columns_to_drop = ["customerID"]

feature_engineering_pipeline = Pipeline(
    [
        ("change_datatypes", ConvertDataTransformer()),
        ("column_dropper", ColumnDropperTransformer(columns_to_drop)),
        (
            "prep",
            ColumnTransformer(
                [
                    (
                        "encode",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        object_columns,
                    )
                ],
                remainder="passthrough",
            ).set_output(transform="pandas"),
        ),
        ("rename_columns", RenameColumns()),
    ]
)