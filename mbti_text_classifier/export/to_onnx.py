from pathlib import Path

import torch
from transformers import AutoTokenizer

from mbti_text_classifier.models.distilbert import DistilBERTClassifier


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return outputs.logits


def export_onnx(cfg, checkpoint_path, output_path):
    model_type = cfg.model_type

    if model_type == "distilbert":
        export_distilbert_onnx(cfg, checkpoint_path, output_path)
    else:
        raise ValueError(f"ONNX export not supported for {model_type}")


def export_distilbert_onnx(cfg, checkpoint_path, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = DistilBERTClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    wrapped_model = ONNXWrapper(model)

    tokenizer = AutoTokenizer.from_pretrained(cfg.models.name)
    dummy_input = tokenizer(
        "This is a sample text for export",
        max_length=cfg.data.tokenizer.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    torch.onnx.export(
        wrapped_model,
        (input_ids, attention_mask),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
    )
