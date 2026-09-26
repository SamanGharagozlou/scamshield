from dataclasses import dataclass, field

from .signals import ScamSignal


@dataclass
class ConversationState:
    conversation_id: str
    transcript: list[str] = field(default_factory=list)
    signals: set[ScamSignal] = field(default_factory=set)
    risk_score: float = 0.0
    risk_level: str = "SAFE"

    def add_text(self, text: str) -> None:
        self.transcript.append(text)

    def add_signals(self, signals: list[ScamSignal]) -> None:
        self.signals.update(signals)

    def recent_context(self, limit: int = 6) -> list[str]:
        return self.transcript[-limit:]

    def reset(self) -> None:
        self.transcript.clear()
        self.signals.clear()
        self.risk_score = 0.0
        self.risk_level = "SAFE"
