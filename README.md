# Customer Support Ticket Classifier

This is a machine learning and deep learning project that classifies customer support tickets from their text. It predicts two things for each ticket:

- **Category:** the department or type of issue the customer is reporting.
- **Priority:** how urgent the ticket is.

Two approaches are built and compared:

1. **Part 1, traditional machine learning:** TF-IDF and Logistic Regression.
2. **Part 2, deep learning and NLP:** DistilBERT, plus an improved TF-IDF baseline tuned with GridSearchCV.

A FastAPI application serves predictions for single tickets and for batches of tickets.

## Project objectives

- Generate a customer support ticket dataset.
- Clean and prepare the dataset for training.
- Split the data into training, validation, and test sets.
- Train machine learning models for category and priority prediction.
- Fine-tune DistilBERT for text classification.
- Compare model performance, training time, and inference speed.
- Build a REST API with FastAPI.
- Test the API with automated tests and Swagger UI.

## Technologies used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF, Logistic Regression, GridSearchCV, and evaluation |
| Hugging Face Transformers | DistilBERT tokenization and fine-tuning |
| PyTorch | Deep learning model training |
| Joblib | Saving and loading machine learning models |
| FastAPI | Building the prediction API |
| Pydantic | Validating API request data |
| Uvicorn | Running the API server |
| Pytest | Automated testing |
| Matplotlib / Seaborn | Generating evaluation plots |

## Dataset

The dataset was generated for this exercise and is stored in `data/tickets.csv`. It has four ticket categories (Account, Billing, Delivery, and Technical Support) and three priority levels (High, Medium, and Low).

The prepared dataset holds 600 tickets, evenly distributed across classes.

| Category | Tickets |
|---|---|
| Account | 150 |
| Billing | 150 |
| Delivery | 150 |
| Technical Support | 150 |
| **Total** | **600** |

| Priority | Tickets |
|---|---|
| High | 200 |
| Medium | 200 |
| Low | 200 |
| **Total** | **600** |

### Dataset splitting

The data was split with stratification into three sets.

| Dataset | Tickets | Used for |
|---|---|---|
| Training | 420 | Training the models |
| Validation | 90 | Evaluating performance during development |
| Testing | 90 | Final evaluation |
| **Total** | **600** | |

## Project structure

```
customer-ticket-classifier/
│
├── data/
│ ├── tickets.csv
│ ├── train.csv
│ ├── validation.csv
│ └── test.csv
│
├── models/
│ ├── category_model.joblib
│ └── priority_model.joblib
│
├── outputs/
│ ├── model_comparison.csv
│ ├── category_confusion_matrix.png
│ ├── priority_confusion_matrix.png
│ └── confusing_tickets.csv
│
├── src/
│ ├── generate_data.py
│ ├── prepare_data.py
│ ├── train_model.py
│ └── train_distilbert.py
│
├── tests/
│ └── test_main.py
│
├── main.py
├── requirements.txt
└── README.md
```

| Path | Description |
|---|---|
| `data/` | The original dataset and the prepared training, validation, and test data |
| `models/category_model.joblib` | Saved machine learning model for category prediction |
| `models/priority_model.joblib` | Saved machine learning model for priority prediction |
| `models/distilbert/category/` | Fine-tuned DistilBERT model and tokenizer for category prediction |
| `models/distilbert/priority/` | Fine-tuned DistilBERT model and tokenizer for priority prediction |
| `outputs/model_comparison.csv` | Measured performance and timing results for the models |
| `outputs/category_confusion_matrix.png` | Category classification results as a confusion matrix |
| `outputs/priority_confusion_matrix.png` | Priority classification results as a confusion matrix |
| `outputs/confusing_tickets.csv` | Confusing or difficult tickets found during analysis |
| `src/generate_data.py` | Generates the original ticket dataset |
| `src/prepare_data.py` | Prepares and splits the dataset |
| `src/train_model.py` | Trains the baseline machine learning models |
| `src/train_distilbert.py` | Trains and evaluates the TF-IDF and DistilBERT models |
| `main.py` | The FastAPI application |
| `tests/test_main.py` | Automated API tests |
| `requirements.txt` | Project dependencies |

The two DistilBERT training folders in `outputs/` are created during training. They are separate from the final fine-tuned models saved under `models/distilbert/`.

## Part 1: Traditional machine learning

### TF-IDF

TF-IDF stands for Term Frequency and Inverse Document Frequency. It turns text into numerical features a machine learning model can use. Words that help tell documents apart get more weight, and words that appear across many documents get less.

### Logistic Regression

Logistic Regression is a supervised algorithm for classification. Here, one Logistic Regression model predicts ticket category and another predicts ticket priority.

### Baseline workflow

1. Generate the ticket dataset.
2. Prepare and split the data.
3. Convert ticket text into TF-IDF features.
4. Train the Logistic Regression models.
5. Evaluate category and priority predictions.
6. Save the trained models with Joblib.
7. Load the models into FastAPI for predictions.

## Part 2: Deep learning and NLP

### DistilBERT

DistilBERT is a smaller, faster version of BERT. It uses a Transformer architecture to learn how words relate to each other and to their surrounding context. Unlike a basic TF-IDF representation, it builds contextual representations of text.

Two separate DistilBERT models are fine-tuned, one for category and one for priority.

### Tokenization

The DistilBERT tokenizer converts ticket text into tokens and numerical input IDs. It also produces attention masks, which tell the model which tokens to attend to. The maximum sequence length is 128 tokens.

### Training configuration

| Parameter | Value |
|---|---|
| Pretrained model | `distilbert/distilbert-base-uncased` |
| Number of epochs | 2 |
| Batch size | 8 |
| Learning rate | 0.00002 |
| Weight decay | 0.01 |
| Maximum token length | 128 |
| Evaluation | After each epoch |

### Improved TF-IDF baseline

Part 2 also tunes the TF-IDF and Logistic Regression models with GridSearchCV. The search covers:

- N-gram ranges `(1, 1)` and `(1, 2)`.
- Logistic Regression regularization parameter `C` values of `0.1`, `1`, and `10`.
- Three-fold cross-validation.
- Macro F1-score as the selection metric.

This gives a tuned traditional baseline to compare against DistilBERT.

## Model evaluation

The models were judged on four measures:

- **Accuracy:** the share of predictions that are correct.
- **Macro F1-score:** the average F1-score across classes, with every class weighted equally.
- **Training time:** how long the model took to train.
- **Inference time per ticket:** the measured average time to make one prediction.

### Measured results

These results were recorded in `outputs/model_comparison.csv`.

| Task | Model | Accuracy | Macro F1 |
|---|---|---|---|
| Category | TF-IDF + Logistic Regression | 1.0000 | 1.0000 |
| Category | DistilBERT | 1.0000 | 1.0000 |
| Priority | TF-IDF + Logistic Regression | 1.0000 | 1.0000 |
| Priority | DistilBERT | 1.0000 | 1.0000 |

### Training time and inference speed

| Task | Model | Training time (seconds) | Inference time per ticket (seconds) |
|---|---|---|---|
| Category | TF-IDF + Logistic Regression | 11.8513 | 0.0001 |
| Category | DistilBERT | 273.2062 | 0.0753 |
| Priority | TF-IDF + Logistic Regression | 13.1851 | 0.0001 |
| Priority | DistilBERT | 277.2822 | 0.0795 |

### Observations

Both approaches reached 100% accuracy and a macro F1-score of 1.00 on the test set. The TF-IDF baseline trained much faster than DistilBERT and had far lower inference time per ticket. DistilBERT needed more training time and computing resources without improving the measured scores. On this dataset, TF-IDF with Logistic Regression is the more efficient choice.

The dataset is synthetic, and its ticket wording gives strong clues about the labels. Perfect scores here don't guarantee similar results on real customer support tickets. Measuring practical generalization would take real, diverse, previously unseen tickets.

## FastAPI application

FastAPI exposes the saved machine learning models through REST endpoints. The application loads the saved category and priority models at startup.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Shows basic application information |
| GET | `/health` | Checks application and model status |
| POST | `/predict` | Predicts category and priority for one ticket |
| POST | `/predict-batch` | Predicts category and priority for multiple tickets |

### Single ticket prediction

```
POST /predict
Content-Type: application/json

{
  "text": "I was charged twice for my order."
}
```

The response includes the predicted category, the predicted priority, and confidence information. The exact values depend on the model's prediction.

### Batch prediction

```
POST /predict-batch
Content-Type: application/json

{
  "tickets": [
    {
      "text": "I was charged twice for my order."
    },
    {
      "text": "My package has not arrived yet."
    },
    {
      "text": "I cannot log in to my account."
    }
  ]
}
```

The endpoint returns predictions for each submitted ticket. If your `TicketRequest` or batch schema uses a different field name, use that instead of the one in this example.

### Input validation

The API validates requests with Pydantic. Empty ticket text is rejected with an HTTP `422 Unprocessable Entity` response.

## Running the project

**1. Install dependencies.** Activate your Python virtual environment, then run:

```bash
pip install -r requirements.txt
```

**2. Generate the dataset:**

```bash
python src/generate_data.py
```

**3. Prepare the dataset:**

```bash
python src/prepare_data.py
```

**4. Train the baseline models:**

```bash
python src/train_model.py
```

This trains the TF-IDF and Logistic Regression models and saves them under `models/`.

**5. Train and compare DistilBERT:**

```bash
python src/train_distilbert.py
```

This fine-tunes the DistilBERT models, evaluates them, and writes the comparison to `outputs/model_comparison.csv`. Training can take several minutes, and the first run downloads the pretrained model. Saved model files can be reused without retraining if they are present and compatible with your code.

**6. Start the FastAPI server** from the project root:

```bash
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs in your browser. Swagger UI lets you try the endpoints interactively. Press `Ctrl + C` in the terminal to stop the server.

## Testing

Automated tests live in `tests/test_main.py`. Run them with:

```bash
python -m pytest tests -v
```

The tests cover the home endpoint, the health endpoint, single ticket prediction, empty ticket text, missing required text, and batch prediction.

### Test results

All six automated tests passed in the earlier test run.

I also launched the application manually, and these requests returned successful HTTP responses:

| Request | Observed result |
|---|---|
| `GET /` | `200 OK` |
| `GET /docs` | `200 OK` |
| `GET /openapi.json` | `200 OK` |
| `POST /predict` | `200 OK` |
| `POST /predict-batch` | `200 OK` |

So the application started and the tested endpoints responded. The logs don't show the exact predictions returned or the response from `/health`.

## Key learnings

Through this project I learned how to:

- Generate and prepare a text classification dataset.
- Convert text into numerical features with TF-IDF.
- Train and evaluate Logistic Regression classifiers.
- Work with tokenization and attention masks.
- Fine-tune a pretrained Transformer model with Hugging Face.
- Tune machine learning hyperparameters with GridSearchCV.
- Compare accuracy, macro F1-score, training time, and inference speed.
- Save and reload trained models.
- Build a REST API with FastAPI and validate its inputs with Pydantic.
- Test API endpoints with Pytest and Swagger UI.

## Conclusion

This project compares two ways of classifying customer support tickets: traditional machine learning with TF-IDF and Logistic Regression, and deep learning with DistilBERT. Both scored perfectly on the synthetic test set, but TF-IDF with Logistic Regression was much faster to train and to run. On the current evaluation, it is the more efficient option for this dataset.

