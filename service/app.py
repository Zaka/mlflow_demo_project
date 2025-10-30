import os
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from service.schemas import PredictRequest, PredictItem, ModelInfo


# --- Config (env vars with sensible defaults) ---
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8082")
MODEL_NAME = os.getenv("MODEL_NAME", "adult-income-xgboost")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    try:
        mv = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)
    except Exception as e:
        raise RuntimeError(f"Could not resolve alias '{MODEL_NAME}@{MODEL_ALIAS}': {e}")

    model_uri = f"models:/{MODEL_NAME}/{mv.version}"  # immutable, reproducible
    # sklearn flavor returns the Pipeline directly
    pipeline = mlflow.sklearn.load_model(model_uri)

    # Cache everything for quick requests
    app.state.model = pipeline  # Full pipeline
    app.state.model_name = MODEL_NAME
    app.state.model_alias = MODEL_ALIAS
    app.state.model_version = mv.version
    app.state.model_run_id = mv.run_id
    app.state.model_source = mv.source
    app.state.final_estimator = pipeline[-1]  # Classifier
    app.state.class_index = {
        int(c): i for i, c in enumerate(app.state.final_estimator.classes_)
    }

    print(
        f"[startup] Loaded {MODEL_NAME}@{MODEL_ALIAS} -> "
        f"version {mv.version} (run_id={mv.run_id}) from {mv.source}"
    )

    yield


app = FastAPI(title="Adult Income Service", lifespan=lifespan)


# Minimal endpoints to verify startup state ---
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_name=app.state.model_name,
        alias=app.state.model_alias,
        version=app.state.model_version,
        run_id=app.state.model_run_id,
        source=app.state.model_source,
    )


@app.post("/predict", response_model=List[PredictItem])
def predict(req: PredictRequest) -> List[PredictItem]:
    df = pd.DataFrame([r.model_dump(by_alias=True) for r in req.records])

    labels = app.state.model.predict(df)
    probs = app.state.model.predict_proba(df)
    idx1 = app.state.class_index[1]

    return [{"prob_1": float(p[idx1]), "label": int(l)} for p, l in zip(probs, labels)]
