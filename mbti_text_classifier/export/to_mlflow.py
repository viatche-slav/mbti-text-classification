from pathlib import Path

import mlflow
import torch

from mbti_text_classifier.models.distilbert import DistilBERTClassifier
from mbti_text_classifier.models.logistic_regression import (
    LogisticRegressionClassifier,
)


def export_to_mlflow(cfg, checkpoint_path, model_name="mbti_classifier"):
    mlflow.set_tracking_uri(cfg.logging.tracking_uri)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_type = cfg.models.type
    with mlflow.start_run(run_name=f"register_{model_name}"):
        if model_type == "distilbert":
            model = DistilBERTClassifier.load_from_checkpoint(checkpoint_path)
            model.eval()
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model.model,
                artifact_path="model",
                registered_model_name=model_name,
                pip_requirements=[
                    f"torch=={torch.__version__}",
                    "transformers~=4.44.0",
                ],
            )
        elif model_type == "logistic_regression":
            model = LogisticRegressionClassifier.load(checkpoint_path)
            model_info = mlflow.sklearn.log_model(
                sk_model=model.model,
                artifact_path="model",
                registered_model_name=model_name,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("checkpoint_path", str(checkpoint_path))

    return model_info
