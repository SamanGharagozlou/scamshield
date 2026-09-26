from scamshield.detector import ScamDetector
from scamshield.models import TranscriptEvent


def test_bank_otp_scam_becomes_critical():
    detector = ScamDetector()

    messages = [
        "Hello, I'm calling from your bank.",
        "We need to fix this immediately.",
        "Please read the code to me.",
    ]

    result = None

    for index, text in enumerate(
        messages,
        start=1,
    ):
        result = detector.process(
            TranscriptEvent(
                conversation_id="bank-scam",
                timestamp=float(index),
                speaker="caller",
                text=text,
            )
        )

    assert result is not None
    assert result.risk_level == "CRITICAL"
    assert result.risk_score >= 80

    assert "BANK_CLAIM" in result.signals
    assert "OTP_REQUEST" in result.signals
    assert "URGENCY" in result.signals

    assert (
        result.scam_category
        == "BANK_IMPERSONATION"
    )

    assert (
        result.recommended_action
        == "DO_NOT_SHARE_CODE"
    )


def test_safe_conversation_stays_safe():
    detector = ScamDetector()

    messages = [
        "Hello, how are you today?",
        "I wanted to check what time dinner is.",
        "See you later this evening.",
    ]

    result = None

    for index, text in enumerate(
        messages,
        start=1,
    ):
        result = detector.process(
            TranscriptEvent(
                conversation_id="safe-call",
                timestamp=float(index),
                speaker="caller",
                text=text,
            )
        )

    assert result is not None

    assert result.risk_level == "SAFE"
    assert result.risk_score < 30
    assert result.signals == []


def test_detector_keeps_conversations_separate():
    detector = ScamDetector()

    detector.process(
        TranscriptEvent(
            conversation_id="conversation-a",
            timestamp=1,
            speaker="caller",
            text="Hello, I'm calling from your bank.",
        )
    )

    result = detector.process(
        TranscriptEvent(
            conversation_id="conversation-b",
            timestamp=1,
            speaker="caller",
            text="Hello, how are you today?",
        )
    )

    assert result.risk_level == "SAFE"
    assert result.risk_score < 30
    assert result.signals == []


def test_detector_reset_removes_previous_state():
    detector = ScamDetector()

    detector.process(
        TranscriptEvent(
            conversation_id="reset-test",
            timestamp=1,
            speaker="caller",
            text="Hello, I'm calling from your bank.",
        )
    )

    detector.reset(
        "reset-test"
    )

    result = detector.process(
        TranscriptEvent(
            conversation_id="reset-test",
            timestamp=2,
            speaker="caller",
            text="Hello, how are you today?",
        )
    )

    assert result.risk_level == "SAFE"
    assert result.risk_score < 30
    assert result.signals == []


def test_detector_returns_ml_score():
    detector = ScamDetector()

    result = detector.process(
        TranscriptEvent(
            conversation_id="ml-score-test",
            timestamp=1,
            speaker="caller",
            text=(
                "Please tell me the verification "
                "code that appeared on your phone."
            ),
        )
    )

    assert result.ml_scam_score is not None

    assert (
        0.0
        <= result.ml_scam_score
        <= 1.0
    )

    assert result.detection_mode == "hybrid"


def test_rules_only_detector_mode():
    detector = ScamDetector(
        use_ml=False
    )

    result = detector.process(
        TranscriptEvent(
            conversation_id="rules-test",
            timestamp=1,
            speaker="caller",
            text="Hello, how are you today?",
        )
    )

    assert result.ml_scam_score is None
    assert result.detection_mode == "rules"
    assert result.risk_score == 0
    assert result.risk_level == "SAFE"
