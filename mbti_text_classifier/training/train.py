from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from mbti_text_classifier.data.download import download_data
from mbti_text_classifier.data.preprocess import preprocess_mbti_dataset
from mbti_text_classifier.models.distilbert import DistilBERTClassifier
from mbti_text_classifier.models.logistic_regression import LogisticRegressionClassifier
from mbti_text_classifier.utils.logging import (
    end_run,
    get_git_commit,
    init_mlflow,
    log_metrics,
    log_params,
    start_run,
)
from mbti_text_classifier.utils.visualization import (
    save_class_distribution,
    save_confusion_matrix,
    save_training_curves,
)


class MBTIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, padding, truncation):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train(cfg):
    pl.seed_everything(cfg.seed)

    init_mlflow(cfg)

    download_data(data_file=cfg.data.data_file, url=cfg.data.download_url)

    data_path = Path(cfg.data.data_file)
    train_df, val_df, test_df = preprocess_mbti_dataset(
        data_path=data_path,
        target_column=cfg.data.target_column,
        train_size=cfg.data.split.train_size,
        val_size=cfg.data.split.val_size,
        test_size=cfg.data.split.test_size,
        random_state=cfg.seed,
    )

    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(cfg.paths.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    class_names = sorted(train_df[cfg.data.target_column].unique())
    class_dist_fig = None
    if cfg.logging.log.artifacts:
        class_dist_fig = save_class_distribution(
            train_df[cfg.data.target_column].tolist(),
            val_df[cfg.data.target_column].tolist(),
            class_names,
            plots_dir / "class_distribution.png",
        )

    if cfg.models.type == "logistic_regression":
        start_run(cfg)

        if class_dist_fig is not None:
            mlflow.log_figure(class_dist_fig, "class_distribution.png")
            plt.close(class_dist_fig)

        if cfg.logging.log.params:
            log_params(
                {
                    "batch_size": cfg.data.batch_size,
                    "max_features": cfg.models.vectorizer.max_features,
                    "ngram_range": str(cfg.models.vectorizer.ngram_range),
                    "min_df": cfg.models.vectorizer.min_df,
                    "max_df": cfg.models.vectorizer.max_df,
                    "max_iter": cfg.models.max_iter,
                    "solver": cfg.models.solver,
                }
            )

        model = LogisticRegressionClassifier(
            num_labels=cfg.data.num_classes,
            max_features=cfg.models.vectorizer.max_features,
            ngram_range=tuple(cfg.models.vectorizer.ngram_range),
            min_df=cfg.models.vectorizer.min_df,
            max_df=cfg.models.vectorizer.max_df,
            max_iter=cfg.models.max_iter,
            solver=cfg.models.solver,
            random_state=cfg.models.random_state,
        )
        model.fit(
            train_df[cfg.data.text_column],
            train_df[cfg.data.target_column],
        )

        if cfg.logging.log.metrics:
            train_metrics = model.evaluate(
                train_df[cfg.data.text_column], train_df[cfg.data.target_column]
            )
            val_metrics = model.evaluate(
                val_df[cfg.data.text_column], val_df[cfg.data.target_column]
            )

            log_metrics(
                {
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1_macro": train_metrics["f1_macro"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision_macro": val_metrics["precision_macro"],
                    "val_recall_macro": val_metrics["recall_macro"],
                    "val_f1_macro": val_metrics["f1_macro"],
                }
            )

            if cfg.logging.log.artifacts:
                val_preds = model.predict(val_df[cfg.data.text_column])
                val_true = val_df[cfg.data.target_column].tolist()
                class_names = sorted(val_df[cfg.data.target_column].unique())

                fig_cm = save_confusion_matrix(
                    val_true,
                    val_preds,
                    class_names,
                    plots_dir / "logistic_regression_confusion_matrix.png",
                )
                mlflow.log_figure(fig_cm, "logistic_regression_confusion_matrix.png")
                plt.close(fig_cm)

            test_metrics = model.evaluate(
                test_df[cfg.data.text_column], test_df[cfg.data.target_column]
            )
            log_metrics(
                {
                    "test_accuracy": test_metrics["accuracy"],
                    "test_precision_macro": test_metrics["precision_macro"],
                    "test_recall_macro": test_metrics["recall_macro"],
                    "test_f1_macro": test_metrics["f1_macro"],
                }
            )

        model.save(model_dir / cfg.models.save_file)
        end_run()

    elif cfg.models.type == "distilbert":
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.experiment_name,
            tracking_uri=cfg.logging.tracking_uri,
            run_name=cfg.run_name,
        )

        if cfg.logging.log.git_commit:
            git_commit = get_git_commit()
            if git_commit:
                mlflow_logger.experiment.set_tag(
                    mlflow_logger.run_id, "git_commit", git_commit
                )

        if class_dist_fig is not None:
            mlflow.log_figure(class_dist_fig, "class_distribution.png")
            plt.close(class_dist_fig)

        tokenizer = AutoTokenizer.from_pretrained(cfg.models.name)

        label_map = {
            label: idx
            for idx, label in enumerate(
                sorted(train_df[cfg.data.target_column].unique())
            )
        }
        train_labels = train_df[cfg.data.target_column].map(label_map)
        val_labels = val_df[cfg.data.target_column].map(label_map)

        train_dataset = MBTIDataset(
            train_df[cfg.data.text_column],
            train_labels,
            tokenizer,
            cfg.data.tokenizer.max_length,
            cfg.data.tokenizer.padding,
            cfg.data.tokenizer.truncation,
        )
        val_dataset = MBTIDataset(
            val_df[cfg.data.text_column],
            val_labels,
            tokenizer,
            cfg.data.tokenizer.max_length,
            cfg.data.tokenizer.padding,
            cfg.data.tokenizer.truncation,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=cfg.data.dataloader.shuffle_train,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=cfg.data.dataloader.shuffle_val,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )

        model = DistilBERTClassifier(
            model_name=cfg.models.name,
            num_labels=cfg.data.num_classes,
            lr=cfg.models.lr,
            weight_decay=cfg.models.weight_decay,
        )

        class MetricsCallback(pl.Callback):
            def __init__(self):
                self.metrics = {
                    "train_loss": [],
                    "train_accuracy": [],
                    "val_loss": [],
                    "val_accuracy": [],
                }

            def on_train_epoch_end(self, trainer, pl_module):
                if "train_loss" in trainer.callback_metrics:
                    self.metrics["train_loss"].append(
                        float(trainer.callback_metrics["train_loss"])
                    )
                if "train_accuracy" in trainer.callback_metrics:
                    self.metrics["train_accuracy"].append(
                        float(trainer.callback_metrics["train_accuracy"])
                    )

            def on_validation_epoch_end(self, trainer, pl_module):
                if "val_loss" in trainer.callback_metrics:
                    self.metrics["val_loss"].append(
                        float(trainer.callback_metrics["val_loss"])
                    )
                if "val_accuracy" in trainer.callback_metrics:
                    self.metrics["val_accuracy"].append(
                        float(trainer.callback_metrics["val_accuracy"])
                    )

        metrics_callback = MetricsCallback()

        trainer = pl.Trainer(
            max_epochs=cfg.training.epochs,
            accelerator=cfg.device,
            logger=mlflow_logger,
            log_every_n_steps=cfg.logging.log_every_n_steps,
            default_root_dir=str(model_dir),
            enable_checkpointing=False,
            callbacks=[metrics_callback],
        )
        trainer.fit(model, train_loader, val_loader)
        trainer.save_checkpoint(model_dir / cfg.models.save_file)

        if cfg.logging.log.artifacts:
            fig_curves = save_training_curves(
                metrics_callback.metrics,
                plots_dir / "training_curves.png",
            )
            mlflow.log_figure(fig_curves, "training_curves.png")
            plt.close(fig_curves)

        if cfg.logging.log.metrics and cfg.logging.log.artifacts:
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(model.device)
                    attention_mask = batch["attention_mask"].to(model.device)
                    labels = batch["labels"]

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1).cpu()

                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())

            class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
            fig_cm = save_confusion_matrix(
                all_labels,
                all_preds,
                class_names,
                plots_dir / "distilbert_confusion_matrix.png",
            )
            mlflow.log_figure(fig_cm, "distilbert_confusion_matrix.png")
            plt.close(fig_cm)

        test_labels = test_df[cfg.data.target_column].map(label_map)
        test_dataset = MBTIDataset(
            test_df[cfg.data.text_column],
            test_labels,
            tokenizer,
            cfg.data.tokenizer.max_length,
            cfg.data.tokenizer.padding,
            cfg.data.tokenizer.truncation,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )

        model.eval()
        test_preds = []
        test_labels_list = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu()

                test_preds.extend(preds.tolist())
                test_labels_list.extend(labels.tolist())

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        test_acc = accuracy_score(test_labels_list, test_preds)
        test_f1 = f1_score(
            test_labels_list, test_preds, average="macro", zero_division=0
        )
        test_precision = precision_score(
            test_labels_list, test_preds, average="macro", zero_division=0
        )
        test_recall = recall_score(
            test_labels_list, test_preds, average="macro", zero_division=0
        )

        mlflow.log_metrics(
            {
                "test_accuracy": test_acc,
                "test_f1_macro": test_f1,
                "test_precision_macro": test_precision,
                "test_recall_macro": test_recall,
            }
        )

    else:
        raise ValueError(f"Unknown model type: {cfg.models.type}")
