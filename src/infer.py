import argparse
import json
import yaml
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def parse_cli_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_xgboost.yaml")
    parser.add_argument("--alias", type=str, default="production")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prediction_input", type=str)
    group.add_argument("--input_path", type=str)

    args = parser.parse_args()

    return args.config, args.alias, args.prediction_input, args.input_path


def audit_alias(model_name: str, alias: str, tracking_uri: str = None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    mv = client.get_model_version_by_alias(name=model_name, alias=alias)

    print(f"[Model alias]: {model_name}@{alias} -> version {mv.version}")
    print(f"[Model run_id]: {mv.run_id}")
    print(f"[Model source]: {mv.source}")

    return mv.version, mv.run_id


def load_model(tracking_uri, model_name, alias):
    version, run_id = audit_alias(model_name, alias, tracking_uri)
    # mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{version}"

    return mlflow.sklearn.load_model(model_uri=model_uri)


def parquet_to_dataframe(path):
    pass


def json_to_dataframe(path):
    pass


def main():
    config, alias, prediction_input, input_path = parse_cli_command()

    cfg = load_config(config)
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    model_cfg = cfg["model"]

    if model_cfg["type"] == "adult-income-xgboost":
        model_name = "adult-income-xgboost"
        model = load_model(tracking_uri, model_name, alias)
    else:
        model_name = "adult-income-lightgbm"
        model = load_model(tracking_uri, model_name, alias)

    if prediction_input:
        df = pd.DataFrame(data=[json.loads(prediction_input)])
    elif input_path:
        if input_path.lower().endswith(".parquet"):
            df = pd.read_parquet(input_path)
        elif input_path.lower().endswith(".json"):
            df = pd.read_json(input_path)
        elif input_path.lower().endswith(".jsonl"):
            df = pd.read_json(input_path, lines=True)            
        elif input_path.lower().endswith(".csv"):
            df = pd.read_csv(input_path)
        else:
            import sys
            print("File type not supported")
            sys.exit(1)

    cols = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    if bool(df[cols].isna().any().any()):
        raise ValueError("Numeric fields contains NaN after coercion.")

    label_indexes = model.steps[-1][1].classes_.tolist()    
    # label_1_idx = [idx for idx in label_indexes if idx == 1][0]
    label_1_idx = label_indexes.index(1)

    labels = model.predict(df)

    probs = model.predict_proba(df)

    output = []

    for p, l in zip(probs, labels):
        prob_1 = float(p[label_1_idx])
        output.append({"prob_1": prob_1, "label": int(l)})

    print(f"output: {json.dumps(output, indent=4)}")    


if __name__ == "__main__":
    main()
