import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as SKPipeline

# from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score

import mlflow
from mlflow.models import infer_signature

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

# SEEDS
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.2


def main():
    df = pd.read_pickle(os.path.join(".", "data", "processed", "dataset.pkl"))

    y = df["class"]
    X = df[[col for col in df.columns if col != "class"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=(1 - TEST_SIZE) * (VALID_SIZE / (1 - TEST_SIZE)),
        random_state=RANDOM_STATE,
        stratify=y_train,
    )  # 0.8 x 0.25 = 0.2

    numeric_features = [
        "age",
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    numeric_transformer = SKPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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

    categorical_transformer = SKPipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    # In a scikit-learn Pipeline, xgb__eval_set is passed straight to 
    # XGBClassifier.fit without going through your preprocessor.

    # So your model trains on OHE-transformed features but validates 
    # on raw features → mismatch → headaches.

    # The clean pattern is: 
    # - fit the preprocessor first
    # - transform train/val
    # - then fit XGBoost with callbacks. 
    # - After training, wrap the fitted pieces back into a pipeline so 
    # your saved model still does end-to-end preprocessing.

    preprocessor.fit(X_train, y_train)

    X_train_tr = preprocessor.transform(X_train)
    X_val_tr = preprocessor.transform(X_val)
    X_test_tr = preprocessor.transform(X_test)

    early_stop = EarlyStopping(rounds=50, save_best=True, maximize=False)

    # Define the model hyperparameters
    params = {
        "n_estimators": 500,  # Few hundreds
        "max_depth": 16,
        "learning_rate": 0.05,  # 0.05-0.3
        "tree_method": "hist",
        "device": "cpu",
        "random_state": RANDOM_STATE,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
        "callbacks": [early_stop],
    }

    xgb = XGBClassifier(**params)

    xgb.fit(
        X_train_tr,
        y_train,
        eval_set=[(X_val_tr, y_val)],
        verbose=True,
    )

    y_pred = xgb.predict(X_test_tr)
    y_proba = xgb.predict_proba(X_test_tr)[:, 1]

    clf = SKPipeline(steps=[("preprocessor", preprocessor), ("xgb", xgb)])

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Configure MLFlow
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8082")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Adult income classifier")

    # Start an MLflow run
    with mlflow.start_run() as run:
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

        features_dict = {
            "run_id": run.info.run_id,
            "raw_columns": list(df.columns),
            "expanded_features": clf.named_steps["preprocessor"]
            .get_feature_names_out()
            .tolist(),
            "readable_features": [
                f.replace("num__", "").replace("cat__", "")
                for f in clf.named_steps["preprocessor"].get_feature_names_out()
            ],
        }

        mlflow.log_dict(features_dict, "feature_names.json")

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
