from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="Customer Support Ticket Classifier",
    description="Predict ticket category and priority",
    version="1.0.0",
)

category_model = None
priority_model = None


@app.on_event("startup")
def load_models():
    global category_model, priority_model

    if (
        os.path.exists("models/category_model.joblib")
        and os.path.exists("models/priority_model.joblib")
    ):
        category_model = joblib.load(
            "models/category_model.joblib"
        )
        priority_model = joblib.load(
            "models/priority_model.joblib"
        )


class TicketRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Customer Support Ticket Classifier API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": (
            category_model is not None
            and priority_model is not None
        ),
    }


@app.post("/predict")
def predict_ticket(ticket: TicketRequest):
    if category_model is None or priority_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Train the models first.",
        )

    text = ticket.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Ticket text cannot be empty.",
        )

    category = category_model.predict([text])[0]
    priority = priority_model.predict([text])[0]

    return {
        "ticket_text": text,
        "predicted_category": str(category),
        "predicted_priority": str(priority),
    }