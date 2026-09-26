from scamshield.risk import calculate_risk
from scamshield.signals import ScamSignal


def test_no_signals_is_safe():
    score, level = calculate_risk(set())

    assert score == 0
    assert level == "SAFE"


def test_bank_claim_alone_is_safe():
    score, level = calculate_risk({
        ScamSignal.BANK_CLAIM,
    })

    assert score == 10
    assert level == "SAFE"


def test_bank_and_urgency_remain_safe():
    score, level = calculate_risk({
        ScamSignal.BANK_CLAIM,
        ScamSignal.URGENCY,
    })

    assert score == 22
    assert level == "SAFE"


def test_otp_mention_with_urgency_is_caution():
    score, level = calculate_risk({
        ScamSignal.BANK_CLAIM,
        ScamSignal.URGENCY,
        ScamSignal.OTP_MENTION,
    })

    assert score == 37
    assert level == "CAUTION"


def test_bank_otp_request_is_critical():
    score, level = calculate_risk({
        ScamSignal.BANK_CLAIM,
        ScamSignal.URGENCY,
        ScamSignal.OTP_REQUEST,
    })

    assert score >= 80
    assert level == "CRITICAL"


def test_remote_access_tech_support_is_critical():
    score, level = calculate_risk({
        ScamSignal.TECH_SUPPORT_CLAIM,
        ScamSignal.REMOTE_ACCESS_REQUEST,
    })

    assert score >= 80
    assert level == "CRITICAL"


def test_score_is_capped_at_100():
    score, level = calculate_risk(set(ScamSignal))

    assert score == 100
    assert level == "CRITICAL"
