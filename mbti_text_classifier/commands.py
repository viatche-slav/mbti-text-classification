from pathlib import Path

import fire
from hydra import compose, initialize_config_dir

from mbti_text_classifier.export.to_mlflow import export_to_mlflow
from mbti_text_classifier.export.to_onnx import export_onnx
from mbti_text_classifier.export.to_tensorrt import export_tensorrt
from mbti_text_classifier.inference.infer import infer
from mbti_text_classifier.serving.app import main as serve_app
from mbti_text_classifier.training.train import train


class Commands:
    @staticmethod
    def _get_config_dir():
        config_dir = Path(__file__).parent.parent / "configs"
        return str(config_dir.absolute())

    def _run_with_config(self, config_name, func, args=None, overrides=None):
        with initialize_config_dir(
            config_dir=self._get_config_dir(), version_base=None
        ):
            overrides_list = []
            if overrides:
                for key, value in overrides.items():
                    overrides_list.append(f"{key}={value}")

            cfg = compose(config_name=config_name, overrides=overrides_list)

            if args:
                return func(cfg, *args)
            return func(cfg)

    def train(self, model_type="distilbert", **overrides):
        overrides["models"] = model_type
        self._run_with_config("train", train, overrides=overrides)

    def infer(self, input_text, model_path=None, **overrides):
        return self._run_with_config(
            "inference",
            infer,
            args=(input_text, model_path),
            overrides=overrides,
        )

    def export_onnx(self, checkpoint_path, output_path, **overrides):
        self._run_with_config(
            "inference",
            export_onnx,
            args=(checkpoint_path, output_path),
            overrides=overrides,
        )

    def export_tensorrt(self, onnx_path, output_path, **overrides):
        self._run_with_config(
            "inference",
            export_tensorrt,
            args=(onnx_path, output_path),
            overrides=overrides,
        )

    def export_mlflow(self, checkpoint_path, model_name="mbti_classifier", **overrides):
        return self._run_with_config(
            "train",
            export_to_mlflow,
            args=(checkpoint_path, model_name),
            overrides=overrides,
        )

    def serve(self, **overrides):
        self._run_with_config("serving", serve_app, overrides=overrides)


def main():
    fire.Fire(Commands)
