from .classifier import SemanticScamClassifier
from .conversation import ConversationState
from .models import TranscriptEvent, ScamRiskEvent
from .risk import calculate_risk
from .rules import detect_signals
from .signals import ScamSignal


class ScamDetector:

    def __init__(
        self,
        use_ml: bool = True,
    ):
        self.sessions: dict[
            str,
            ConversationState
        ] = {}

        self.use_ml = use_ml

        self.classifier = (
            SemanticScamClassifier()
            if use_ml
            else None
        )

    def process(
        self,
        event: TranscriptEvent,
    ) -> ScamRiskEvent:

        if event.conversation_id not in self.sessions:

            self.sessions[
                event.conversation_id
            ] = ConversationState(
                conversation_id=event.conversation_id
            )

        state = self.sessions[
            event.conversation_id
        ]

        # 1. Rule-based detection on the new transcript chunk.
        detected = detect_signals(
            event.text
        )

        state.add_text(event.text)
        state.add_signals(detected)

        # 2. Build rolling conversation context.
        recent_context = " ".join(
            state.recent_context(limit=6)
        )

        # 3. Semantic ML scam score.
        ml_scam_score = None

        if self.classifier is not None:
            ml_scam_score = (
                self.classifier.predict_score(
                    recent_context
                )
            )

        # 4. Hybrid risk calculation.
        score, level = calculate_risk(
            state.signals,
            ml_scam_score=ml_scam_score,
        )

        state.risk_score = score
        state.risk_level = level

        category = None
        action = None

        if ScamSignal.BANK_CLAIM in state.signals:
            category = "BANK_IMPERSONATION"

        if (
            ScamSignal.TECH_SUPPORT_CLAIM
            in state.signals
        ):
            category = "TECH_SUPPORT_SCAM"

        if (
            ScamSignal.GOVERNMENT_CLAIM
            in state.signals
        ):
            category = "GOVERNMENT_IMPERSONATION"

        if (
            ScamSignal.FAMILY_EMERGENCY
            in state.signals
        ):
            category = "FAMILY_EMERGENCY_SCAM"

        if ScamSignal.OTP_REQUEST in state.signals:
            action = "DO_NOT_SHARE_CODE"

        elif (
            ScamSignal.PASSWORD_REQUEST
            in state.signals
        ):
            action = "DO_NOT_SHARE_PASSWORD"

        elif ScamSignal.PIN_REQUEST in state.signals:
            action = "DO_NOT_SHARE_PIN"

        elif (
            ScamSignal.MONEY_TRANSFER_REQUEST
            in state.signals
        ):
            action = "DO_NOT_SEND_MONEY"

        elif (
            ScamSignal.GIFT_CARD_REQUEST
            in state.signals
        ):
            action = "DO_NOT_BUY_GIFT_CARDS"

        elif (
            ScamSignal.REMOTE_ACCESS_REQUEST
            in state.signals
        ):
            action = "DO_NOT_ALLOW_REMOTE_ACCESS"

        return ScamRiskEvent(
            conversation_id=event.conversation_id,
            timestamp=event.timestamp,
            risk_score=score,
            risk_level=level,
            signals=sorted(
                signal.value
                for signal in state.signals
            ),
            scam_category=category,
            recommended_action=action,
            ml_scam_score=(
                round(ml_scam_score, 4)
                if ml_scam_score is not None
                else None
            ),
            detection_mode=(
                "hybrid"
                if self.use_ml
                else "rules"
            ),
        )

    def reset(
        self,
        conversation_id: str,
    ) -> None:
        self.sessions.pop(
            conversation_id,
            None,
        )
