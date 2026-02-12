from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from transformers import AutoTokenizer

from mbti_text_classifier.models.distilbert import DistilBERTClassifier


def create_app(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    checkpoint_path = Path(cfg.model.checkpoint_path)
    try:
        model = DistilBERTClassifier.load_from_checkpoint(checkpoint_path)
        model.eval()
        error_msg = None
    except Exception as error:
        model = None
        error_msg = f"Failed to load model: {str(error)}"
        print(f"ERROR: {error_msg}")

    app = FastAPI(title="MBTI Text Classifier")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "mbti_types": cfg.mbti_types,
                "error": error_msg if model is None else None,
            },
        )

    @app.post("/predict", response_class=HTMLResponse)
    async def predict(request: Request, text: str = Form(...)):
        if model is None:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "mbti_types": cfg.mbti_types,
                    "error": error_msg,
                },
            )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=cfg.model.max_length,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.softmax(logits, dim=1)[0]

        predicted_idx = int(torch.argmax(probs))
        mbti_types = list(cfg.mbti_types)
        predicted_type = mbti_types[predicted_idx]
        confidence = float(probs[predicted_idx])

        top_3_idx = torch.argsort(probs, descending=True)[:3].tolist()
        top_3 = [
            {"type": mbti_types[idx], "probability": f"{probs[idx]:.3f}"}
            for idx in top_3_idx
        ]

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "mbti_types": mbti_types,
                "result": {
                    "type": predicted_type,
                    "confidence": f"{confidence:.3f}",
                    "top_3": top_3,
                },
            },
        )

    return app


def main(cfg: DictConfig = None):
    if cfg is None:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        with initialize_config_dir(
            config_dir=str(config_dir.absolute()), version_base=None
        ):
            cfg = compose(config_name="serving")

    app = create_app(cfg)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")


if __name__ == "__main__":
    main()
