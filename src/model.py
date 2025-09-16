import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score

import mlflow
from mlflow.models import infer_signature

from xgboost import XGBClassifier

# SEEDS
RANDOM_STATE = 42


def main():
    df = pd.read_pickle(os.path.join(".", "data", "processed", "dataset.pkl"))

    y = df["class"]
    X = df[[col for col in df.columns if col != "class"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=RANDOM_STATE, stratify=y
    )

    numeric_features = [
        "age",
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder='passthrough'
    )

    # Define the model hyperparameters
    params = {
        "n_estimators": 500,  # Few hundreds
        "max_depth": 16,
        "learning_rate": 0.05,  # 0.05-0.3
        "tree_method": "hist",
        "device": "cuda",
        "random_state": RANDOM_STATE,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", XGBClassifier(**params))]
    )

    # bst = XGBClassifier(**params)

    # bst.fit(X_train, y_train, verbose=True)
    clf.fit(X_train, y_train, classifier__verbose=True)

    # y_pred = bst.predict(X_test)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Configure MLFlow
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8082")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Adult income classifier")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        )

        # Infer the model signature
        signature = infer_signature(X_train, clf.predict(X_train))

        # Log the model, which inherits the parameters and metric
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            name="adult_income",
            signature=signature,
            input_example=X_train,
            registered_model_name="xgboost",
        )

        # Set a tag that we can use to remind ourselves what this model was for
        mlflow.set_logged_model_tags(
            model_info.model_id,
            {"Training Info": "Basic XGBoost classifier for adult income"},
        )


if __name__ == "__main__":
    main()
