from scamshield.risk import calculate_risk
from scamshield.signals import ScamSignal


def test_no_signals_is_safe():
    score, level = calculate_risk(
        set()
    )

    assert score == 0
    assert level == "SAFE"


def test_bank_claim_alone_is_safe():
    score, level = calculate_risk(
        {
            ScamSignal.BANK_CLAIM,
        }
    )

    assert score == 10
    assert level == "SAFE"


def test_bank_and_urgency_remain_safe():
    score, level = calculate_risk(
        {
            ScamSignal.BANK_CLAIM,
            ScamSignal.URGENCY,
        }
    )

    assert score == 22
    assert level == "SAFE"


def test_otp_mention_with_urgency_is_caution():
    score, level = calculate_risk(
        {
            ScamSignal.BANK_CLAIM,
            ScamSignal.URGENCY,
            ScamSignal.OTP_MENTION,
        }
    )

    assert score == 37
    assert level == "CAUTION"


def test_bank_otp_request_is_critical():
    score, level = calculate_risk(
        {
            ScamSignal.BANK_CLAIM,
            ScamSignal.URGENCY,
            ScamSignal.OTP_REQUEST,
        }
    )

    assert score >= 80
    assert level == "CRITICAL"


def test_remote_access_tech_support_is_critical():
    score, level = calculate_risk(
        {
            ScamSignal.TECH_SUPPORT_CLAIM,
            ScamSignal.REMOTE_ACCESS_REQUEST,
        }
    )

    assert score >= 80
    assert level == "CRITICAL"


def test_score_is_capped_at_100():
    score, level = calculate_risk(
        {
            ScamSignal.BANK_CLAIM,
            ScamSignal.GOVERNMENT_CLAIM,
            ScamSignal.TECH_SUPPORT_CLAIM,
            ScamSignal.FAMILY_EMERGENCY,
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
    )

    assert score == 100
    assert level == "CRITICAL"


def test_ml_threshold_becomes_high():
    score, level = calculate_risk(
        set(),
        ml_scam_score=0.40,
    )

    assert score >= 50
    assert level == "HIGH"


def test_below_ml_threshold_not_high():
    score, level = calculate_risk(
        set(),
        ml_scam_score=0.39,
    )

    assert score < 50
    assert level == "CAUTION"


def test_ml_alone_does_not_become_critical():
    score, level = calculate_risk(
        set(),
        ml_scam_score=1.0,
    )

    assert score == 75.0
    assert level == "HIGH"


def test_low_ml_score_remains_safe():
    score, level = calculate_risk(
        set(),
        ml_scam_score=0.3381,
    )

    assert score < 30
    assert level == "SAFE"


def test_medium_ml_score_becomes_caution():
    score, level = calculate_risk(
        set(),
        ml_scam_score=0.37,
    )

    assert 30 <= score < 50
    assert level == "CAUTION"


def test_ml_plus_danger_signal_can_escalate():
    score, level = calculate_risk(
        {
            ScamSignal.REMOTE_ACCESS_REQUEST,
        },
        ml_scam_score=0.70,
    )

    assert score > 50
    assert level in {
        "HIGH",
        "CRITICAL",
    }
