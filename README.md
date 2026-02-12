# MBTI Text Classification

Определение психотипа по тексту постов на основе типологии Майерс-Бриггс (MBTI).

## Описание проекта

### Постановка задачи

Модель предсказывает MBTI-тип личности человека по тексту его постов в социальных сетях. Это позволяет автоматизировать анализ личности пользователей для построения персонализированных рекомендаций.

### Входные и выходные данные

**Вход:** Текст из 50 последних постов пользователя, объединенных разделителем `|||`

**Выход:** Один из 16 MBTI-типов, комбинация из 4 осей:

- Introversion (I) / Extroversion (E)
- Intuition (N) / Sensing (S)
- Thinking (T) / Feeling (F)
- Judging (J) / Perceiving (P)

### Метрики

Задача многоклассовой классификации (16 классов). Основные метрики:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

### Данные

Используется [MBTI Myers-Briggs Personality Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type) (~9000 записей).

Разделение: 70% train / 15% validation / 15% test (stratified split).

### Модели

**Baseline:** Logistic Regression с TF-IDF векторизацией

**Основная модель:** DistilBERT (fine-tuned на задаче классификации)

---

## Setup

### Требования

- Python 3.11
- Poetry

### Установка

```bash
python3.11 -m venv venv
source venv/bin/activate
poetry install
pre-commit install
pre-commit run -a
```

---

## Train

### MLflow Tracking

Перед обучением надо запустить MLflow UI:

```bash
poetry run mlflow server --host 127.0.0.1 --port 8080
```

После чего открыть http://127.0.0.1:8080 для просмотра метрик и артефактов.

### Обучение DistilBERT

```bash
poetry run mbti-cli train --model_type=distilbert training.epochs=5 seed=123
```

### Обучение Logistic Regression

```bash
poetry run mbti-cli train --model_type=logistic_regression
```

### Артефакты после обучения

После успешного обучения создаются:

- **Модель**: `models/distilbert.ckpt` или `models/logistic_regression.pkl`
- **Графики**:
  - `plots/class_distribution.png` - распределение классов
  - `plots/training_curves.png` - кривые обучения (loss/accuracy)
  - `plots/distilbert_confusion_matrix.png` - матрица ошибок
- **MLflow**: эксперименты в `mlruns/`, артефакты в `mlartifacts/`

---

## Production Preparation

### Экспорт в ONNX

```bash
poetry run mbti-cli export_onnx models/distilbert.ckpt models/distilbert.onnx
```

### Экспорт в TensorRT

```bash
poetry run mbti-cli export_tensorrt models/distilbert.onnx models/distilbert.trt
```

### Комплектация поставки

Минимальный набор для inference:

```
mbti_text_classifier/
├── models/
│   ├── distilbert.py
│   └── logistic_regression.py
├── inference/
│   └── infer.py
└── configs/
    ├── config.yaml
    ├── data/mbti.yaml
    └── models/distilbert.yaml

models/distilbert.ckpt  # Обученная модель
```

---

## Inference

### CLI Inference

**DistilBERT:**

```bash
poetry run mbti-cli infer "Some post."
```

**С явным указанием модели:**

```bash
poetry run mbti-cli infer "Text here" --model_path=models/distilbert.ckpt
```

**Logistic Regression:**

```bash
poetry run mbti-cli infer "Text here" --model_type=logistic_regression
```

### Web Server

Надо запустить веб-интерфейс для inference:

```bash
poetry run mbti-cli serve
```

После чего открыть http://0.0.0.0:8000 в браузере.

**Настройка модели** в `configs/serving.yaml`:

```yaml
model:
  checkpoint_path: models/distilbert.ckpt
```
