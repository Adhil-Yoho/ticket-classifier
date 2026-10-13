
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report,)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

required_columns = ["text", "category", "priority"]

for df in [train_df, val_df, test_df]:
    if not all(column in df.columns for column in required_columns):
        raise ValueError("A dataset is missing required columns")

print("Training rows:", len(train_df))
print("Validation rows", len(val_df))
print("Testing rows", len(test_df))

def train_classifier(label, model_file, matrix_file):
    print("\n" + "="*50)
    print("Training", label, "model")
    print("="*50)

    X_train = train_df["text"]
    y_train = train_df[label]

    X_val = val_df["text"]
    y_val = val_df[label]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    parameters = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "model__C": [0.1, 1, 10],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("Best parameters:", grid.best_params_)

    predictions = best_model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro", zero_division=0,
    )

    print("Validation accuracy:", round(accuracy, 4))
    print("Validation macro F1:", round(macro_f1, 4))

    print("\nClassification report:")
    print(classification_report(y_val, predictions, zero_division=0,))

    labels = best_model.named_steps["model"].classes_

    cm = confusion_matrix(
        y_val, predictions, labels=labels
    )

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{label.title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(matrix_file)
    plt.close()

    print("Confusion matrix saved to:", matrix_file)

    joblib.dump(best_model, model_file)
    print("Model saved to:", model_file)

    y_true = y_val.to_numpy()
    wrong_predictions = predictions != y_true

    errors = val_df.loc[
        wrong_predictions, ["text"]
    ].copy()

    errors["actual"] = y_true[wrong_predictions]
    errors["predicted"] = predictions[wrong_predictions]
    errors["label_type"] = label

    print("Misclassified validation tickets:", len(errors))
    return best_model, errors

category_model, category_errors = train_classifier(
    label="category",
    model_file="models/category_model.joblib",
    matrix_file="outputs/category_confusion_matrix.png",
)

priority_model, priority_errors = train_classifier(
    label="priority",
    model_file="models/priority_model.joblib",
    matrix_file="outputs/priority_confusion_matrix.png",
)

all_errors = pd.concat(
    [category_errors, priority_errors],
    ignore_index=True,
)

error_file = "outputs/confusing_tickets.csv"
all_errors.to_csv(error_file, index=False)

print("\n Confusing tickets results saved to:", error_file)


print("\nFinal test results")

for label, model in [
    ("category", category_model),
    ("priority", priority_model),
]:
    test_predictions = model.predict(test_df["text"])
    accuracy = accuracy_score(
        test_df[label],
        test_predictions,
    )
    macro_f1 = f1_score(
        test_df[label],
        test_predictions,
        average="macro",
        zero_division=0,
    )

    print("\n" + label.title())
    print("Test accuracy:", round(accuracy, 4))
    print("Test macro F1:", round(macro_f1, 4))
        


print("\n Ten most confusing examples:")
if all_errors.empty:
    print("No misclassified tickets were found")
else:
    print(all_errors[["text","label_type","actual","predicted"]
        ].head(10).to_string(index=False))
    
print("\n Training and evaluation completed")