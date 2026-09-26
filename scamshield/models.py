from pydantic import BaseModel


class TranscriptEvent(BaseModel):
    conversation_id: str
    timestamp: float
    speaker: str
    text: str


class ScamRiskEvent(BaseModel):
    conversation_id: str
    timestamp: float
    risk_score: float
    risk_level: str
    signals: list[str]
    scam_category: str | None = None
    recommended_action: str | None = None
