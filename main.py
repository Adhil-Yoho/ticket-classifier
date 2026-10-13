from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os

category_model = None
priority_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global category_model, priority_model

    category_path = "models/category_model.joblib"
    priority_path = "models/priority_model.joblib"

    if os.path.exists(category_path) and os.path.exists(priority_path):
        category_model = joblib.load(category_path)
        priority_model = joblib.load(priority_path)

    yield

app = FastAPI(
    title="Customer Support Ticket Classifier",
    description="Predict ticket category and priority",
    version="2.0.0",
    lifespan=lifespan,
)


class TicketRequest(BaseModel):
    text: str = Field(min_length=1)

class BatchTicketRequest(BaseModel):
    tickets: list[TicketRequest] = Field(min_length=1)

def predict_one(text):
    if category_model is None or priority_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Train the models first.",
        )

    text = text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Ticket text cannot be empty.",
        )

    category = category_model.predict([text])[0]
    priority = priority_model.predict([text])[0]

    category_probabilities = category_model.predict_proba([text])[0]
    priority_probabilities = priority_model.predict_proba([text])[0]

    category_confidence = float(max(category_probabilities))
    priority_confidence = float(max(priority_probabilities))

    confidence = (
        category_confidence + priority_confidence
    ) / 2

    return {
        "category": str(category),
        "priority": str(priority),
        "confidence": round(confidence, 4),
        "category_confidence": round(category_confidence, 4),
        "priority_confidence": round(priority_confidence, 4),
    }


@app.get("/")
def home():
    return {
        "message": "Customer Support Ticket Classifier API"
    }

@app.get("/health")
def health():
    loaded = (
        category_model is not None
        and priority_model is not None
    )

    return {
        "status": "ok" if loaded else "models_not_loaded",
        "models_loaded": loaded,
    }


@app.post("/predict")
def predict_ticket(ticket: TicketRequest):
    return predict_one(ticket.text)

@app.post("/predict-batch")
def predict_batch(request: BatchTicketRequest):
    results = [
        predict_one(ticket.text)
        for ticket in request.tickets
    ]

    return {"predictions": results}
