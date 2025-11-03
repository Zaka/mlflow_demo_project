import argparse
import json
import tempfile
import yaml
import pandas as pd
import mlflow

from src.utils import die

from mlflow.tracking import MlflowClient


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def parse_cli_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_xgboost.yaml")
    parser.add_argument("--alias", type=str, default="production")
    group = parser.add_mutually_exclusive_group(required=True)
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

    try:
        return mlflow.sklearn.load_model(model_uri=model_uri), run_id
    except Exception as e:
        die(20, "Model loading failed", str(e))


def load_feature_spec(client, run_id):
    tmp = tempfile.mkdtemp()
    local = client.download_artifacts(run_id, "feature_names.json", tmp)
    with open(local, "r") as f:
        spec = json.load(f)

    return spec


def main():
    config, alias, prediction_input, input_path = parse_cli_command()

    cfg = load_config(config)
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    model_cfg = cfg["model"]

    if model_cfg["type"] == "adult-income-xgboost":
        model_name = "adult-income-xgboost"
        model, run_id = load_model(tracking_uri, model_name, alias)
    else:
        model_name = "adult-income-lightgbm"
        model, run_id = load_model(tracking_uri, model_name, alias)

    if prediction_input:
        obj = json.loads(prediction_input)
        df = pd.DataFrame(obj if isinstance(obj, list) else [obj])
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
            die(2, "File type not supported", f"input_path={input_path}")

    client = MlflowClient()
    spec = load_feature_spec(client, run_id)
    target = spec.get("target", "class")
    required_raw = [c for c in spec["raw_columns"] if c != target]

    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        die(10, "Missing required columns", detail=str(sorted(missing)))

    df = df[required_raw]

    cols = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]

    try:
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    except Exception as e:
        die(20, "Failed when coercing numeric columns.", str(e))

    if bool(df[cols].isna().any().any()):
        die(10, "Numeric fields contain NaN after coercion.")

    label_indexes = model.steps[-1][1].classes_.tolist()
    # label_1_idx = [idx for idx in label_indexes if idx == 1][0]
    label_1_idx = label_indexes.index(1)

    labels = model.predict(df)

    probs = model.predict_proba(df)

    output = []

    for p, l in zip(probs, labels):
        prob_1 = float(p[label_1_idx])
        output.append({"prob_1": prob_1, "label": int(l)})

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
