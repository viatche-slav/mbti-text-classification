import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mbti_text_classifier.models.distilbert import DistilBERTClassifier
from mbti_text_classifier.models.logistic_regression import LogisticRegressionClassifier


def infer(cfg, input_text, model_path=None):
    if model_path is None:
        model_path = Path(cfg.model_dir) / cfg.models.save_file

    model_type = cfg.model_type

    if model_type == "logistic_regression":
        result = infer_logistic_regression(cfg, input_text, model_path)
    elif model_type == "distilbert":
        result = infer_distilbert(cfg, input_text, model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if cfg.output_format == "json":
        return json.dumps(result, indent=2)
    return result


def infer_logistic_regression(cfg, input_text, model_path):
    model = LogisticRegressionClassifier.load(model_path)
    prediction = model.predict(input_text)
    proba = model.predict_proba(input_text)[0]

    class_names = sorted(cfg.data.class_names)
    confidence = max(proba)

    if confidence < cfg.confidence_threshold:
        return {
            "prediction": None,
            "confidence": float(confidence),
            "reason": "below_threshold",
        }

    probabilities = {class_names[i]: float(proba[i]) for i in range(len(class_names))}

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "probabilities": probabilities,
    }


def infer_distilbert(cfg, input_text, model_path):
    if cfg.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = DistilBERTClassifier.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.models.name)
    inputs = tokenizer(
        input_text,
        max_length=cfg.data.tokenizer.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        prediction_idx = torch.argmax(probs).item()
        confidence = probs[prediction_idx].item()

    class_names = sorted(cfg.data.class_names)

    if confidence < cfg.confidence_threshold:
        return {
            "prediction": None,
            "confidence": confidence,
            "reason": "below_threshold",
        }

    prediction = class_names[prediction_idx]
    probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
    }
