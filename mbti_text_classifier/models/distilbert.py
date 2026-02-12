import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification


class DistilBERTClassifier(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, weight_decay, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        if self.class_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = criterion(outputs.logits, batch["labels"])

        preds = torch.argmax(outputs.logits, dim=1)

        self.training_step_outputs.append(
            {
                "preds": preds,
                "labels": batch["labels"],
                "loss": loss,
            }
        )

        return loss

    def on_train_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.training_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.training_step_outputs])
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()

        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()

        acc = (all_preds == all_labels).mean()
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        self.log("train_loss", avg_loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        self.log("train_f1_macro", f1)

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        if self.class_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = criterion(outputs.logits, batch["labels"])

        preds = torch.argmax(outputs.logits, dim=1)

        self.validation_step_outputs.append(
            {
                "preds": preds,
                "labels": batch["labels"],
                "loss": loss,
            }
        )

        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()

        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()

        acc = (all_preds == all_labels).mean()
        precision = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        self.log("val_precision_macro", precision)
        self.log("val_recall_macro", recall)
        self.log("val_f1_macro", f1, prog_bar=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
