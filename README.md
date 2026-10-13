# Customer Support Ticket Classifier

This project sorts customer support tickets by two labels:

- **Category:** Billing, Technical Support, Account, or Delivery
- **Priority:** Low, Medium, or High

The baseline models are TF-IDF with Logistic Regression, and a FastAPI app serves predictions through a REST API.

## Dataset

The dataset is synthetic, made for this learning project. It has 600 ticket examples, each with a unique text, in three columns:

| Column | Description |
|---|---|
| `text` | Customer support ticket description |
| `category` | Ticket category |
| `priority` | Ticket urgency level |

### Class distribution

Category:

| Class | Count |
|---|---|
| Billing | 150 |
| Technical Support | 150 |
| Account | 150 |
| Delivery | 150 |

Priority:

| Class | Count |
|---|---|
| Low | 200 |
| Medium | 200 |
| High | 200 |

### Data source

`src/generate_data.py` generates the data locally. It does not represent real customer interactions.

## Data preprocessing

- Removed rows with missing text or labels.
- Converted ticket text to lowercase and stripped leading and trailing whitespace.
- Removed duplicate ticket texts.
- Checked the class distributions for both labels.
- Split the data into training (70%), validation (15%), and test (15%) sets, stratified on the combined category-priority label.

## Model development

There are two separate pipelines, one for category and one for priority. Each uses TF-IDF with Logistic Regression.

`GridSearchCV` tunes the TF-IDF n-gram range and the Logistic Regression regularization parameter `C`, using three-fold cross-validation on the training data.

## Evaluation

Each model is checked with:

- Accuracy
- Macro F1-score
- Classification report
- Confusion matrix
- Examples of misclassified tickets

Validation metrics drive model selection, and the test set is kept for the final evaluation. To get the actual results, run `python -m src.train_model`.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Check API and model status |
| POST | `/predict` | Predict category and priority |

When the server is running, interactive documentation is available at `/docs`.

## Project structure

- `src/generate_data.py`: generates the dataset
- `src/prepare_data.py`: cleans and splits the data
- `src/train_model.py`: tunes, trains, and evaluates the models
- `main.py`: the FastAPI application
- `models/`: saved model pipelines
- `outputs/`: confusion matrices and error-analysis results

## How to run

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate the dataset, prepare the splits, and train the models:

```bash
python -m src.generate_data
python -m src.prepare_data
python -m src.train_model
```

Start the API:

```bash
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/docs` to test predictions.

## Limitations

The dataset is synthetic and its wording is deliberately informative. High scores here do not show how the models would perform on real tickets. Before any practical use, evaluate them on a representative real-world dataset.
