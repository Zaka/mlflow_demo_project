import argparse
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score

import mlflow
from mlflow.models import infer_signature

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from lightgbm import LGBMClassifier, early_stopping

import shap

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def load_dataset():

    path = os.path.join(".", "data", "processed", "dataset.pkl")

    try:
        df = pd.read_pickle(path)
    except FileNotFoundError:
        from src.data.create_dataset import load_dataset

        load_dataset()
        df = pd.read_pickle(path)

    return df


def partition_dataset(X, y, test_size, val_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=(1 - test_size) * (val_size / (1 - test_size)),
        random_state=seed,
        stratify=y_train,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_conf_matrix_fig(y_test, y_pred, labels, mlflow, normalize=None):
    # Log the confusion matrix as a figure
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize=normalize)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    if normalize is None:
        value_format = "d"
        title = "Confusion Matrix (Test)"
        fig_log_name = "plots/confusion_matrix_test.png"
    elif normalize == "true":
        value_format = ".2f"
        title = "Confusion Matrix (Normalized, Test)"
        fig_log_name = "plots/confusion_matrix_test_normalized.png"

    disp.plot(ax=ax, colorbar=False, values_format=value_format)
    ax.set_title(title)
    plt.tight_layout()
    mlflow.log_figure(fig, fig_log_name)
    plt.close(fig)

    return cm


def log_conf_matrix_figs(y_test, y_pred, labels, mlflow):
    cm = create_conf_matrix_fig(y_test, y_pred, labels, mlflow)
    cm_norm = create_conf_matrix_fig(y_test, y_pred, labels, mlflow, normalize="true")

    mlflow.log_dict(
        {
            "labels": list(map(str, labels)),
            "matrix": cm.tolist(),
            "matrix_normalized": cm_norm.tolist(),
        },
        "artifacts/confusion_matrix.json",
    )

def log_shap_summary_plot(mlflow, X, explanation, feature_names, test_partition=True):
    plt.figure()  # start a fresh figure so it doesn't collide with other matplotlib stuff
    shap.summary_plot(
        explanation.values,
        X,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()

    if test_partition:
        mlflow.log_figure(plt.gcf(), "plots/shap_summary_test.png")
    else:
        mlflow.log_figure(plt.gcf(), "plots/shap_summary_val.png")

    plt.close()

def log_shap_waterfall_plot(mlflow, y_pred, explanation):
    # pick one row that the model predicted as class 1 (>50K)
    pos_indices = np.where(y_pred == 1)[0]
    if len(pos_indices) == 0:
        # fallback: just take the first row
        idx = 0
    else:
        idx = int(pos_indices[0])
    
    single_expl = explanation[idx]

    plt.figure()
    shap.plots.waterfall(single_expl, show=False)
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), f"plots/shap_waterfall_idx_{idx}.png")
    plt.close()

def log_topk_attributions_table(mlflow, explanation, X, y_pred, k=10):
    pos_indices = np.where(y_pred == 1)[0]
    if len(pos_indices) == 0:
        idx = 0
    else:
        idx = int(pos_indices[0])

    shap_row = explanation.values[idx]
    sample_features = X.iloc[idx]
    names = sample_features.index.tolist()

    # sort features by absolute impact
    order = np.argsort(np.abs(shap_row))[::-1][:k]

    topk = []
    for j in order:
        topk.append({
            "feature": names[j],
            "value": sample_features.iloc[j],
            "shap_value": float(shap_row[j])
        })

    # log as JSON artifact
    mlflow.log_dict(
        {
            "sample_index": int(idx),
            "predicted_class": int(y_pred[idx]),
            "topk": topk,
        },
        "artifacts/topk_attributions_sample.json"
    )

def log_shap_values_and_metadata(mlflow, explanation, feature_names):
    # Save SHAP data for later inspection
    np.save("shap_values.npy", explanation.values)
    np.save("shap_base_values.npy", explanation.base_values)

    with open("shap_feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # Log them as MLflow artifacts
    mlflow.log_artifact("shap_values.npy", "artifacts")
    mlflow.log_artifact("shap_base_values.npy", "artifacts")
    mlflow.log_artifact("shap_feature_names.json", "artifacts")

def log_shap_dependence_plot(mlflow, explanation, X, feature_names):
    for feature_to_plot in ["hours-per-week", "education_Bachelors", "age"]:
        plt.figure()
        shap.dependence_plot(
            feature_to_plot,
            explanation.values,
            X,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), f"plots/shap_dependence_{feature_to_plot}.png")
        plt.close()

def log_shap_global_importance_bar(mlflow, X, explanation, feature_names):
    plt.figure()
    shap.summary_plot(
        explanation.values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "plots/shap_global_importance_bar.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_xgboost.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = cfg["seed"]

    random.seed(seed)
    np.random.seed(seed)

    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]

    dataset = load_dataset()

    y = dataset["class"]
    X = dataset[[col for col in dataset.columns if col != "class"]]

    X_train, X_val, X_test, y_train, y_val, y_test = partition_dataset(
        X, y, test_size, val_size, seed
    )

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
        steps=[("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    # In a scikit-learn Pipeline, xgb__eval_set is passed straight to
    # XGBClassifier.fit without going through the preprocessor.

    # So the model trains on OHE-transformed features but validates
    # on raw features → mismatch → headaches.

    # The clean pattern to:
    # - fit the preprocessor first
    # - transform train/val
    # - then fit XGBoost with callbacks.
    # - After training, wrap the fitted pieces back into a pipeline so
    # the saved model still does end-to-end preprocessing.

    preprocessor.set_output(transform="pandas")
    preprocessor.fit(X_train, y_train)

    X_train_tr = preprocessor.transform(X_train)
    X_val_tr = preprocessor.transform(X_val)
    X_test_tr = preprocessor.transform(X_test)

    # Define the model hyperparameters
    model_cfg = cfg["model"]

    if model_cfg["type"] == "adult-income-xgboost":
        early_stop = EarlyStopping(
            rounds=model_cfg["early_stopping_rounds"], save_best=True, maximize=False
        )

        params = {
            "n_estimators": model_cfg["n_estimators"],
            "max_depth": model_cfg["max_depth"],
            "learning_rate": model_cfg["learning_rate"],
            "tree_method": "hist",
            "device": "cpu",
            "random_state": seed,
            "objective": model_cfg["objective"],
            "eval_metric": "logloss",
            "callbacks": [early_stop],
        }

        booster = XGBClassifier(**params)
    else:  # LightGBM
        early_stop = early_stopping(model_cfg["stopping_rounds"], verbose=True)

        params = {
            "n_estimators": model_cfg["n_estimators"],
            "num_leaves": model_cfg["num_leaves"],  # ~31-63
            "min_child_samples": model_cfg["min_child_samples"],  # ~20-100
            "min_split_gain": model_cfg["min_split_gain"],
            "feature_pre_filter": model_cfg["feature_pre_filter"],
            "learning_rate": model_cfg["learning_rate"],
            "random_state": seed,
            "verbose": model_cfg["verbose"],
            "objective": model_cfg["objective"],
            "num_boost_round": model_cfg["num_boost_round"]
        }

        booster = LGBMClassifier(**params)

    booster.fit(
        X_train_tr,
        y_train,
        eval_set=[(X_val_tr, y_val)],
    )

    # Refit Train + Val #############################################
    
    if model_cfg["type"] == "adult-income-xgboost":
        best_iteration = booster.best_iteration
    else: # LightGBM
        # best_iteration = booster._best_iteration
        best_iteration = booster.booster_.best_iteration

    if best_iteration is None or best_iteration < 1:
        best_iteration = booster.n_estimators

    X_train = pd.concat((X_train, X_val))
    y_train = pd.concat((y_train, y_val))

    preprocessor.fit(X_train, y_train)

    X_train_tr = preprocessor.transform(X_train)

    booster_refit = sklearn.base.clone(booster)

    booster_refit.n_estimators = best_iteration
    booster_refit.callbacks = None
    
    booster_refit.fit(X_train_tr, y_train)

    clf = SKPipeline(steps=[("preprocessor", preprocessor), ("booster", booster_refit)])

    y_pred = booster_refit.predict(X_test_tr)
    y_proba = booster_refit.predict_proba(X_test_tr)[:, 1]
    
    explainer = shap.TreeExplainer(booster_refit)
    explanation_val = explainer(X_val_tr)
    explanation_test = explainer(X_test_tr)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Labels for confusion matrix
    labels = [0, 1]

    mlflow_cfg = cfg["mlflow"]

    # Configure MLFlow
    mlflow.set_tracking_uri(uri=mlflow_cfg["tracking_uri"])

    # Create a new MLflow Experiment
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    # Start an MLflow run
    with mlflow.start_run() as run:
        log_conf_matrix_figs(y_test, y_pred, labels, mlflow)

        feature_names = [
            f.replace("num__", "").replace("cat__", "")
            for f in clf.named_steps["preprocessor"].get_feature_names_out()
        ]

        log_shap_summary_plot(mlflow, X_val_tr, explanation_val, feature_names, 
                            test_partition=False)
        log_shap_summary_plot(mlflow, X_test_tr, explanation_test, feature_names)
        log_shap_dependence_plot(mlflow, explanation_test, X_test_tr, feature_names)
        log_shap_waterfall_plot(mlflow, y_pred, explanation_test)
        log_shap_global_importance_bar(mlflow, X_test_tr, explanation_test, feature_names)
        log_shap_values_and_metadata(mlflow, explanation_test, feature_names)
        log_topk_attributions_table(mlflow, explanation_test, X_test_tr, y_pred)

        if model_cfg["type"] == "adult-income-xgboost":
            mlflow.log_params(
                {
                    "n_estimators": model_cfg["n_estimators"],
                    "max_depth": model_cfg["max_depth"],
                    "learning_rate": model_cfg["learning_rate"],
                    "tree_method": "hist",
                    "device": "cpu",
                    "random_state": seed,
                    "objective": model_cfg["objective"],
                    "eval_metric": "logloss",
                }
            )
        else:  # LightGBM
            mlflow.log_params(
                {
                    "n_estimators": model_cfg["n_estimators"],
                    "num_leaves": model_cfg["num_leaves"],  # ~31-63
                    "num_boost_round": model_cfg["num_boost_round"],
                    "min_child_samples": model_cfg["min_child_samples"],  # ~20-100
                    "min_split_gain": model_cfg["min_split_gain"],
                    "feature_pre_filter": model_cfg["feature_pre_filter"],
                    "learning_rate": model_cfg["learning_rate"],
                    "device": "cpu",
                    "random_state": seed,
                    "verbose": model_cfg["verbose"],
                    "objective": model_cfg["objective"],
                }
            )

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
            "raw_columns": list(dataset.columns),
            "expanded_features": clf.named_steps["preprocessor"]
            .get_feature_names_out()
            .tolist(),
            "readable_features": [
                f.replace("num__", "").replace("cat__", "")
                for f in clf.named_steps["preprocessor"].get_feature_names_out()
            ],
        }

        mlflow.log_dict(features_dict, "feature_names.json")

        mlflow.log_artifact(args.config)
        mlflow.log_dict(cfg, "config_used.yaml")

        X_train_sample = X_train.head()

        for col in numeric_features:
            X_train_sample.loc[:, col] = X_train_sample[col].astype(float)

        # Infer the model signature
        signature = infer_signature(X_train_sample, clf.predict_proba(X_train_sample))

        print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(f"SKPipeline clf: {clf}")

        # Log the model, which inherits the parameters and metric
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            name="adult_income",
            signature=signature,
            input_example=X_train.head(50),
            registered_model_name=model_cfg["type"],
        )

        # Set a tag that we can use to remind ourselves what this model was for
        mlflow.set_tag(
            model_info.model_id,
            {"Training Info": f"Basic {model_cfg["type"]} classifier for adult income"},
        )

        with open("README.txt", "r") as f:
            mlflow.log_text(f.read(), "README.txt")

if __name__ == "__main__":
    main()
