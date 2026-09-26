from .conversation import ConversationState
from .models import TranscriptEvent, ScamRiskEvent
from .risk import calculate_risk
from .rules import detect_signals
from .signals import ScamSignal


class ScamDetector:
    def __init__(self):
        self.sessions: dict[str, ConversationState] = {}

    def process(self, event: TranscriptEvent) -> ScamRiskEvent:
        if event.conversation_id not in self.sessions:
            self.sessions[event.conversation_id] = ConversationState(
                conversation_id=event.conversation_id
            )

        state = self.sessions[event.conversation_id]

        detected = detect_signals(event.text)

        state.add_text(event.text)
        state.add_signals(detected)

        score, level = calculate_risk(state.signals)

        state.risk_score = score
        state.risk_level = level

        category = None
        action = None

        if ScamSignal.BANK_CLAIM in state.signals:
            category = "BANK_IMPERSONATION"

        if ScamSignal.TECH_SUPPORT_CLAIM in state.signals:
            category = "TECH_SUPPORT_SCAM"

        if ScamSignal.GOVERNMENT_CLAIM in state.signals:
            category = "GOVERNMENT_IMPERSONATION"

        if ScamSignal.FAMILY_EMERGENCY in state.signals:
            category = "FAMILY_EMERGENCY_SCAM"

        if ScamSignal.OTP_REQUEST in state.signals:
            action = "DO_NOT_SHARE_CODE"
        elif ScamSignal.PASSWORD_REQUEST in state.signals:
            action = "DO_NOT_SHARE_PASSWORD"
        elif ScamSignal.PIN_REQUEST in state.signals:
            action = "DO_NOT_SHARE_PIN"
        elif ScamSignal.MONEY_TRANSFER_REQUEST in state.signals:
            action = "DO_NOT_SEND_MONEY"
        elif ScamSignal.GIFT_CARD_REQUEST in state.signals:
            action = "DO_NOT_BUY_GIFT_CARDS"
        elif ScamSignal.REMOTE_ACCESS_REQUEST in state.signals:
            action = "DO_NOT_ALLOW_REMOTE_ACCESS"

        return ScamRiskEvent(
            conversation_id=event.conversation_id,
            timestamp=event.timestamp,
            risk_score=score,
            risk_level=level,
            signals=sorted(
                signal.value for signal in state.signals
            ),
            scam_category=category,
            recommended_action=action,
        )

    def reset(self, conversation_id: str) -> None:
        self.sessions.pop(conversation_id, None)
