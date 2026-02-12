import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class LogisticRegressionClassifier:
    def __init__(
        self,
        num_labels,
        max_features,
        ngram_range,
        min_df,
        max_df,
        max_iter,
        solver,
        random_state,
    ):
        self.num_labels = num_labels
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )
        self.model = LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
        )
        self.label_encoder = None
        self.label_decoder = None

    def fit(self, x_train, y_train):
        unique_labels = sorted(set(y_train))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        y_train_encoded = np.array([self.label_encoder[label] for label in y_train])

        x_train_vec = self.vectorizer.fit_transform(x_train)
        self.model.fit(x_train_vec, y_train_encoded)

    def predict(self, x):
        if isinstance(x, str):
            x = [x]
        x_vec = self.vectorizer.transform(x)
        y_pred_encoded = self.model.predict(x_vec)
        y_pred = [self.label_decoder[idx] for idx in y_pred_encoded]

        return y_pred[0] if len(y_pred) == 1 else y_pred

    def predict_proba(self, x):
        if isinstance(x, str):
            x = [x]
        x_vec = self.vectorizer.transform(x)

        return self.model.predict_proba(x_vec)

    def evaluate(self, x, y_true):
        y_pred = self.predict(x)
        y_true_encoded = np.array([self.label_encoder[label] for label in y_true])
        y_pred_encoded = np.array([self.label_encoder[label] for label in y_pred])

        return {
            "accuracy": accuracy_score(y_true_encoded, y_pred_encoded),
            "precision_macro": precision_score(
                y_true_encoded, y_pred_encoded, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true_encoded, y_pred_encoded, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_true_encoded, y_pred_encoded, average="macro", zero_division=0
            ),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "vectorizer": self.vectorizer,
                    "label_encoder": self.label_encoder,
                    "label_decoder": self.label_decoder,
                    "num_labels": self.num_labels,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance.vectorizer = data["vectorizer"]
        instance.label_encoder = data["label_encoder"]
        instance.label_decoder = data["label_decoder"]
        instance.num_labels = data["num_labels"]

        return instance
