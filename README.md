# Customer Support Ticket Classifier

This project sorts customer support tickets by category and by priority, using Natural Language Processing (NLP) and Machine Learning. TF-IDF with Logistic Regression is the baseline model, and FastAPI serves the trained models through a REST API.

- **Category classification:** predicts what kind of issue a ticket is about.
- **Priority classification:** predicts how urgent the ticket is.

## Project objectives

- Create a dataset of at least 500 customer support tickets.
- Clean and prepare the text data.
- Check the class distributions for category and priority.
- Split the data into training, validation, and test sets using stratification.
- Build separate TF-IDF and Logistic Regression pipelines for category and priority.
- Tune hyperparameters with GridSearchCV and cross-validation.
- Evaluate the models with accuracy, macro F1-score, classification reports, and confusion matrices.
- Analyze misclassified ticket examples.
- Save the trained models with joblib.
- Build a FastAPI application that predicts ticket category and priority.

## Dataset

The dataset was generated locally with `src/generate_data.py`. It holds synthetic support tickets written for this learning project, so it contains no real customer interactions. It was built from predefined ticket descriptions and priority-related phrases, not downloaded from an external source.

| Property | Value |
|---|---|
| Total rows | 600 |
| Total columns | 3 |
| Unique ticket texts | 600 |
| Missing text or labels after cleaning | 0 in the generated dataset |
| Format | CSV |

| Column | Description |
|---|---|
| `text` | Description of the customer support issue |
| `category` | Category of the ticket |
| `priority` | Priority level of the ticket |

Both labels are balanced.

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
| Low | 200 |
| Medium | 200 |
| **Total** | **600** |

## Data preprocessing

`src/prepare_data.py` prepares the data. It does the following:

1. Loads the CSV with Pandas.
2. Removes rows with missing text, category, or priority.
3. Converts ticket text to lowercase.
4. Strips leading and trailing whitespace.
5. Removes duplicate ticket texts.
6. Checks the class distributions for both labels.
7. Splits the data into training, validation, and test sets.

| Dataset | Share | Tickets |
|---|---|---|
| Training | 70% | 420 |
| Validation | 15% | 90 |
| Test | 15% | 90 |
| **Total** | **100%** | **600** |

The split uses a fixed random state, so results are reproducible. Stratification is based on the combined category and priority labels, which keeps each combination represented across the three sets.

The script writes three files:

- `data/train.csv`
- `data/validation.csv`
- `data/test.csv`

## Model development

I built two separate models, because category and priority are different prediction targets. Both use the same pipeline: a TF-IDF vectorizer followed by Logistic Regression.

**Category classifier.** Predicts one of Account, Billing, Delivery, or Technical Support.

**Priority classifier.** Predicts one of Low, Medium, or High.

TF-IDF turns text into numbers a machine learning algorithm can use. It weights each word by how often it appears in a ticket and by how common it is across all tickets. Logistic Regression is a supervised classification algorithm that learns the link between those text features and the target labels.

## Hyperparameter tuning

GridSearchCV picked the hyperparameters for each pipeline, using three-fold cross-validation on the training set and macro F1-score as the metric.

| Parameter | Values tested |
|---|---|
| `tfidf__ngram_range` | (1, 1), (1, 2) |
| `model__C` | 0.1, 1, 10 |

Best settings found:

| Model | Best C | Best n-gram range |
|---|---|---|
| Category | 0.1 | (1, 1) |
| Priority | 0.1 | (1, 2) |

A unigram is one word and a bigram is two consecutive words. The parameter `C` controls regularization in Logistic Regression, and smaller values mean stronger regularization.

## Evaluation results

The validation set was used during model selection. The test set was kept back for the final evaluation.

### Category classification

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 1.0000 | 1.0000 |
| Macro F1-score | 1.0000 | 1.0000* |

Validation classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Account | 1.00 | 1.00 | 1.00 | 23 |
| Billing | 1.00 | 1.00 | 1.00 | 22 |
| Delivery | 1.00 | 1.00 | 1.00 | 22 |
| Technical Support | 1.00 | 1.00 | 1.00 | 23 |
| Accuracy | | | 1.00 | 90 |
| Macro average | 1.00 | 1.00 | 1.00 | 90 |

Best parameters: `C = 0.1`, `ngram_range = (1, 1)`.

\*The category test macro F1-score is inferred from the reported test accuracy of 1.00. It assumes every test prediction was correct and that the score was taken over the same four classes. Check the macro F1-score printed by the script before treating this value as directly reported.

### Priority classification

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 1.0000 | 1.0000 |
| Macro F1-score | 1.0000 | 1.0000 |

Validation classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| High | 1.00 | 1.00 | 1.00 | 30 |
| Low | 1.00 | 1.00 | 1.00 | 29 |
| Medium | 1.00 | 1.00 | 1.00 | 31 |
| Accuracy | | | 1.00 | 90 |
| Macro average | 1.00 | 1.00 | 1.00 | 90 |

Best parameters: `C = 0.1`, `ngram_range = (1, 2)`.

### How to read these results

Both models scored 100% accuracy on the validation set and on the test set, so every example in those sets was classified correctly. The dataset is synthetic, though, and its wording strongly signals the category and priority. These scores show that the baseline pipeline works on the generated data. They say little about how well it would do on real customer tickets.

## Confusion matrices

A confusion matrix compares actual labels with predicted labels. Rows are the actual classes, columns are the predicted classes, the diagonal holds correct predictions, and anything off the diagonal is a misclassification.

Two matrices were generated:

- `outputs/category_confusion_matrix.png`
- `outputs/priority_confusion_matrix.png`

Neither model misclassified a validation ticket, so each matrix should show counts only on the diagonal and zeros everywhere else.

## Error analysis

I compared the validation predictions with the actual labels to find misclassified tickets.

| Model | Misclassified validation tickets |
|---|---|
| Category | 0 |
| Priority | 0 |

With no errors, there were no examples to list among the ten most confusing tickets. The error-analysis output is saved as `outputs/confusing_tickets.csv`.

Real support tickets would likely be harder. They often contain:

- Ambiguous or incomplete descriptions
- Words that appear in more than one category
- Misspellings and grammatical errors
- Urgency that is never stated outright
- Several issues in one ticket

Testing on a representative real-world dataset would show how the models handle these.

## Saved models

The trained pipelines are saved with joblib.

| File | Contents |
|---|---|
| `models/category_model.joblib` | Trained category classification pipeline |
| `models/priority_model.joblib` | Trained priority classification pipeline |

Each file holds the fitted TF-IDF vectorizer and the Logistic Regression model. The FastAPI application loads them at startup, so nothing needs retraining.

## FastAPI application

FastAPI exposes the trained models over HTTP.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Returns the API welcome message |
| GET | `/health` | Reports API status and whether both models are loaded |
| POST | `/predict` | Predicts ticket category and priority |
| GET | `/docs` | Opens the interactive API documentation |

Example request: send a POST to `/predict` with this JSON body.

```json
{
  "text": "I cannot log in to my account and this is urgent."
}
```

The API returns the submitted text along with the predicted category and priority. The exact predictions depend on the loaded models.

### API testing

The server logs showed the following:

- The home endpoint returned HTTP 200.
- The health endpoint returned HTTP 200.
- The interactive documentation loaded.
- The prediction endpoint returned HTTP 200 for a test request.

So the application started and handled the requests I tested.

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
│ ├── category_confusion_matrix.png
│ ├── priority_confusion_matrix.png
│ └── confusing_tickets.csv
│
├── src/
│ ├── __init__.py
│ ├── generate_data.py
│ ├── prepare_data.py
│ └── train_model.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and usage

Run these commands from the project root.

**1. Create and activate the virtual environment** (Windows PowerShell):

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

**2. Install dependencies:**

```bash
python -m pip install -r requirements.txt
```

**3. Generate the dataset:**

```bash
python -m src.generate_data
```

**4. Clean and split the dataset:**

```bash
python -m src.prepare_data
```

**5. Train and evaluate the models:**

```bash
python -m src.train_model
```

**6. Start the FastAPI server:**

```bash
uvicorn main:app --reload
```

**7. Test the API.** Open http://127.0.0.1:8000/docs in a browser and use the interactive documentation to try the `/predict` endpoint.

## Technologies used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- GridSearchCV
- Matplotlib
- Seaborn
- Joblib
- FastAPI
- Uvicorn
- Git and GitHub

## Conclusion

This project is a complete baseline workflow for support ticket classification. It covers synthetic data generation, preprocessing, tuning, evaluation, model saving, and a FastAPI service. Separate TF-IDF and Logistic Regression pipelines handle category and priority, and both scored perfectly on the validation and test sets of the synthetic data. It gives you a simple base to build on with more features and API endpoints.
