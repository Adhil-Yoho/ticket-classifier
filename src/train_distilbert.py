import os
import time
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = "distilbert/distilbert-base-uncased"

os.makedirs("models/distilbert", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")


class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_to_id):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        self.labels = [
            label_to_id[label] for label in labels
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(
                self.encodings["input_ids"][index]
            ),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][index]
            ),
            "labels": torch.tensor(
                self.labels[index], dtype=torch.long
            ),
        }


def calculate_metrics(prediction):
    logits, labels = prediction
    predicted_labels = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(
            labels, predicted_labels
        ),
        "macro_f1": f1_score(
            labels, predicted_labels, average="macro"
        ),
    }


def train_and_compare(label):
    print(f"\nTraining DistilBERT for {label}")

    classes = sorted(train_df[label].unique())
    label_to_id = {
        name: index for index, name in enumerate(classes)
    }
    id_to_label = {
        index: name for name, index in label_to_id.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TicketDataset(
        train_df["text"], train_df[label],
        tokenizer, label_to_id,
    )
    val_dataset = TicketDataset(
        val_df["text"], val_df[label],
        tokenizer, label_to_id,
    )
    test_dataset = TicketDataset(
        test_df["text"], test_df[label],
        tokenizer, label_to_id,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(classes),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    output_dir = f"models/distilbert/{label}"

    args = TrainingArguments(
        output_dir=f"outputs/distilbert_{label}",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        seed=42,
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=calculate_metrics,
    )

    start = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - start

    val_result = trainer.evaluate(val_dataset)

    print("Validation accuracy:", round(
        val_result["eval_accuracy"], 4
    ))
    print("Validation macro F1:", round(
        val_result["eval_macro_f1"], 4
    ))

    start = time.perf_counter()
    test_result = trainer.predict(test_dataset)
    inference_seconds = time.perf_counter() - start

    test_predictions = np.argmax(
        test_result.predictions, axis=1
    )
    test_labels = np.array(test_dataset.labels)

    test_accuracy = accuracy_score(
        test_labels, test_predictions
    )
    test_macro_f1 = f1_score(
        test_labels, test_predictions, average="macro"
    )

    print("Test accuracy:", round(test_accuracy, 4))
    print("Test macro F1:", round(test_macro_f1, 4))
    print("Training time (seconds):", round(
        training_seconds, 2
    ))
    print("Inference time per ticket (seconds):", round(
        inference_seconds / len(test_dataset), 5
    ))

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "label": label,
        "model": "DistilBERT",
        "accuracy": test_accuracy,
        "macro_f1": test_macro_f1,
        "training_time_seconds": training_seconds,
        "inference_time_per_ticket_seconds":
            inference_seconds / len(test_dataset),
    }


def train_baseline(label):
    print(f"\nBenchmarking baseline for {label}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    parameters = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "model__C": [0.1, 1, 10],
    }

    grid = GridSearchCV(
        pipeline,
        parameters,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
    )

    start = time.perf_counter()
    grid.fit(train_df["text"], train_df[label])
    training_seconds = time.perf_counter() - start

    best_model = grid.best_estimator_

    joblib_path = f"models/{label}_model.joblib"
    import joblib
    joblib.dump(best_model, joblib_path)

    start = time.perf_counter()
    predictions = best_model.predict(test_df["text"])
    inference_seconds = time.perf_counter() - start

    return {
        "label": label,
        "model": "TF-IDF + Logistic Regression",
        "accuracy": accuracy_score(
            test_df[label], predictions
        ),
        "macro_f1": f1_score(
            test_df[label], predictions, average="macro"
        ),
        "training_time_seconds": training_seconds,
        "inference_time_per_ticket_seconds":
            inference_seconds / len(test_df),
    }


results = []

for target in ["category", "priority"]:
    results.append(train_baseline(target))
    results.append(train_and_compare(target))

results_df = pd.DataFrame(results)
results_df.to_csv(
    "outputs/model_comparison.csv", index=False
)

print("\nFinal model comparison")
print(results_df.round(4).to_string(index=False))
print("\nComparison saved to outputs/model_comparison.csv")
