from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .detector import ScamDetector
from .models import TranscriptEvent


app = FastAPI(
    title="ScamShield Detection API",
    version="0.1.0",
    description="Real-time scam detection backend for ScamShield.",
)

# Demo/development configuration.
# We will restrict this to the real frontend domain before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = ScamDetector()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "scamshield-detection",
    }


@app.post("/detect")
def detect(event: TranscriptEvent):
    return detector.process(event)


@app.delete("/conversations/{conversation_id}")
def reset_conversation(conversation_id: str):
    detector.reset(conversation_id)

    return {
        "status": "reset",
        "conversation_id": conversation_id,
    }
