from .signals import ScamSignal


WEIGHTS = {
    ScamSignal.BANK_CLAIM: 10,
    ScamSignal.GOVERNMENT_CLAIM: 10,
    ScamSignal.TECH_SUPPORT_CLAIM: 10,
    ScamSignal.FAMILY_EMERGENCY: 15,

    ScamSignal.OTP_MENTION: 15,
    ScamSignal.OTP_REQUEST: 35,
    ScamSignal.PASSWORD_REQUEST: 40,
    ScamSignal.PIN_REQUEST: 40,

    ScamSignal.MONEY_TRANSFER_REQUEST: 35,
    ScamSignal.GIFT_CARD_REQUEST: 45,
    ScamSignal.REMOTE_ACCESS_REQUEST: 45,

    ScamSignal.URGENCY: 12,
    ScamSignal.THREAT: 20,
    ScamSignal.SECRECY: 20,
}


def semantic_bonus(ml_scam_score: float | None) -> float:
    """
    Convert the ML classifier score into supporting risk evidence.

    The ML model supports the rule engine, but does not independently
    create a CRITICAL alert.
    """

    if ml_scam_score is None:
        return 0.0

    if ml_scam_score >= 0.90:
        return 35.0

    if ml_scam_score >= 0.75:
        return 25.0

    if ml_scam_score >= 0.60:
        return 15.0

    if ml_scam_score >= 0.50:
        return 8.0

    return 0.0


def calculate_risk(
    signals: set[ScamSignal],
    ml_scam_score: float | None = None,
) -> tuple[float, str]:

    score = sum(
        WEIGHTS.get(signal, 0)
        for signal in signals
    )

    # Dangerous contextual combinations.
    if (
        ScamSignal.BANK_CLAIM in signals
        and ScamSignal.OTP_REQUEST in signals
    ):
        score += 25

    if (
        ScamSignal.TECH_SUPPORT_CLAIM in signals
        and ScamSignal.REMOTE_ACCESS_REQUEST in signals
    ):
        score += 30

    if (
        ScamSignal.GOVERNMENT_CLAIM in signals
        and ScamSignal.THREAT in signals
        and ScamSignal.MONEY_TRANSFER_REQUEST in signals
    ):
        score += 30

    if (
        ScamSignal.FAMILY_EMERGENCY in signals
        and ScamSignal.URGENCY in signals
        and ScamSignal.MONEY_TRANSFER_REQUEST in signals
    ):
        score += 30

    # Add semantic ML evidence.
    score += semantic_bonus(ml_scam_score)

    score = min(float(score), 100.0)

    if score >= 80:
        level = "CRITICAL"
    elif score >= 50:
        level = "HIGH"
    elif score >= 30:
        level = "CAUTION"
    else:
        level = "SAFE"

    return score, level
