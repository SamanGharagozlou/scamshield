from .classifier import SemanticScamClassifier
from .conversation import ConversationState
from .models import TranscriptEvent, ScamRiskEvent
from .risk import calculate_risk
from .rules import detect_signals
from .signals import ScamSignal


# These represent actual suspicious BEHAVIOR,
# rather than merely the caller claiming an identity.
BEHAVIORAL_SIGNALS = {
    ScamSignal.OTP_MENTION,
    ScamSignal.OTP_REQUEST,
    ScamSignal.PASSWORD_REQUEST,
    ScamSignal.PIN_REQUEST,
    ScamSignal.MONEY_TRANSFER_REQUEST,
    ScamSignal.GIFT_CARD_REQUEST,
    ScamSignal.REMOTE_ACCESS_REQUEST,
    ScamSignal.URGENCY,
    ScamSignal.THREAT,
    ScamSignal.SECRECY,
}


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

        # ------------------------------------------------
        # 1. Get/create conversation state
        # ------------------------------------------------

        if (
            event.conversation_id
            not in self.sessions
        ):
            self.sessions[
                event.conversation_id
            ] = ConversationState(
                conversation_id=(
                    event.conversation_id
                )
            )

        state = self.sessions[
            event.conversation_id
        ]

        # ------------------------------------------------
        # 2. Rule detection on current transcript chunk
        # ------------------------------------------------

        detected = detect_signals(
            event.text
        )

        state.add_text(
            event.text
        )

        state.add_signals(
            detected
        )

        # ------------------------------------------------
        # 3. Build rolling context
        # ------------------------------------------------

        recent_context = " ".join(
            state.recent_context(
                limit=6
            )
        )

        # ------------------------------------------------
        # 4. ML classifier
        # ------------------------------------------------

        ml_scam_score = None

        if self.classifier is not None:
            ml_scam_score = (
                self.classifier.predict_score(
                    recent_context
                )
            )

        # ------------------------------------------------
        # 5. Hybrid risk calculation
        # ------------------------------------------------

        score, level = calculate_risk(
            state.signals,
            ml_scam_score=ml_scam_score,
        )

        # ------------------------------------------------
        # 6. LIVE SAFETY GATE
        # ------------------------------------------------
        #
        # The classifier was calibrated primarily on
        # complete conversations.
        #
        # In live mode, identity/topic words like:
        #
        #   "I'm calling from your bank"
        #
        # can produce an elevated ML score before the
        # caller has actually done anything dangerous.
        #
        # Therefore ML evidence alone cannot trigger
        # HIGH until ScamShield also sees suspicious
        # behavioral evidence.
        #
        # This reduces early false alarms.
        # ------------------------------------------------

        has_behavioral_evidence = any(
            signal in BEHAVIORAL_SIGNALS
            for signal in state.signals
        )

        if (
            self.use_ml
            and not has_behavioral_evidence
            and score >= 50
        ):
            score = 49.0
            level = "CAUTION"

        # ------------------------------------------------
        # 7. Store current risk
        # ------------------------------------------------

        state.risk_score = score
        state.risk_level = level

        # ------------------------------------------------
        # 8. Scam category
        # ------------------------------------------------

        category = None

        if (
            ScamSignal.BANK_CLAIM
            in state.signals
        ):
            category = (
                "BANK_IMPERSONATION"
            )

        if (
            ScamSignal.TECH_SUPPORT_CLAIM
            in state.signals
        ):
            category = (
                "TECH_SUPPORT_SCAM"
            )

        if (
            ScamSignal.GOVERNMENT_CLAIM
            in state.signals
        ):
            category = (
                "GOVERNMENT_IMPERSONATION"
            )

        if (
            ScamSignal.FAMILY_EMERGENCY
            in state.signals
        ):
            category = (
                "FAMILY_EMERGENCY_SCAM"
            )

        # ------------------------------------------------
        # 9. Recommended protective action
        # ------------------------------------------------

        action = None

        if (
            ScamSignal.OTP_REQUEST
            in state.signals
        ):
            action = (
                "DO_NOT_SHARE_CODE"
            )

        elif (
            ScamSignal.PASSWORD_REQUEST
            in state.signals
        ):
            action = (
                "DO_NOT_SHARE_PASSWORD"
            )

        elif (
            ScamSignal.PIN_REQUEST
            in state.signals
        ):
            action = (
                "DO_NOT_SHARE_PIN"
            )

        elif (
            ScamSignal.MONEY_TRANSFER_REQUEST
            in state.signals
        ):
            action = (
                "DO_NOT_SEND_MONEY"
            )

        elif (
            ScamSignal.GIFT_CARD_REQUEST
            in state.signals
        ):
            action = (
                "DO_NOT_BUY_GIFT_CARDS"
            )

        elif (
            ScamSignal.REMOTE_ACCESS_REQUEST
            in state.signals
        ):
            action = (
                "DO_NOT_ALLOW_REMOTE_ACCESS"
            )

        # ------------------------------------------------
        # 10. Return API event
        # ------------------------------------------------

        return ScamRiskEvent(
            conversation_id=(
                event.conversation_id
            ),
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
                round(
                    ml_scam_score,
                    4,
                )
                if ml_scam_score
                is not None
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
