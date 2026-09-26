from .signals import ScamSignal


# Calibrated using separate external calibration data.
#
# This is an operational decision threshold.
# It does NOT mean that 0.40 literally represents
# a calibrated "40% probability of scam".
ML_SCAM_THRESHOLD = 0.40

# Below this ML score, the ML component remains
# in the SAFE risk range.
ML_CAUTION_THRESHOLD = 0.35


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


HIGH_DANGER_SIGNALS = {
    ScamSignal.OTP_REQUEST,
    ScamSignal.PASSWORD_REQUEST,
    ScamSignal.PIN_REQUEST,
    ScamSignal.MONEY_TRANSFER_REQUEST,
    ScamSignal.GIFT_CARD_REQUEST,
    ScamSignal.REMOTE_ACCESS_REQUEST,
}


def calculate_rule_score(
    signals: set[ScamSignal],
) -> float:
    """
    Calculate scam risk using deterministic rules only.
    """

    score = sum(
        WEIGHTS.get(signal, 0)
        for signal in signals
    )

    # Bank impersonation + OTP request.
    if (
        ScamSignal.BANK_CLAIM in signals
        and ScamSignal.OTP_REQUEST in signals
    ):
        score += 25

    # Tech-support impersonation + remote-access request.
    if (
        ScamSignal.TECH_SUPPORT_CLAIM in signals
        and ScamSignal.REMOTE_ACCESS_REQUEST in signals
    ):
        score += 30

    # Government impersonation + threat + money request.
    if (
        ScamSignal.GOVERNMENT_CLAIM in signals
        and ScamSignal.THREAT in signals
        and ScamSignal.MONEY_TRANSFER_REQUEST in signals
    ):
        score += 30

    # Family emergency + urgency + money request.
    if (
        ScamSignal.FAMILY_EMERGENCY in signals
        and ScamSignal.URGENCY in signals
        and ScamSignal.MONEY_TRANSFER_REQUEST in signals
    ):
        score += 30

    return min(
        float(score),
        100.0,
    )


def model_score_to_risk(
    ml_scam_score: float,
) -> float:
    """
    Convert the raw ML classifier score into ScamShield's
    operational 0-100 risk scale.

    Mapping:

        ML < 0.35
            -> risk 0-29
            -> SAFE

        ML 0.35-0.40
            -> risk 30-49
            -> CAUTION

        ML >= 0.40
            -> risk 50-75
            -> HIGH

    ML alone is capped at 75.

    Therefore ML by itself cannot produce CRITICAL.
    CRITICAL generally requires corroborating rule evidence.
    """

    score = max(
        0.0,
        min(
            float(ml_scam_score),
            1.0,
        ),
    )

    # SAFE range:
    #
    # ML 0.00 -> risk 0
    # ML 0.35 -> risk just below 30
    if score < ML_CAUTION_THRESHOLD:
        return (
            score
            / ML_CAUTION_THRESHOLD
        ) * 29.0

    # CAUTION range:
    #
    # ML 0.35 -> risk 30
    # ML 0.40 -> risk just below 50
    if score < ML_SCAM_THRESHOLD:
        return 30.0 + (
            (
                score
                - ML_CAUTION_THRESHOLD
            )
            / (
                ML_SCAM_THRESHOLD
                - ML_CAUTION_THRESHOLD
            )
        ) * 19.0

    # HIGH range:
    #
    # ML 0.40 -> risk 50
    # ML 1.00 -> risk 75
    return 50.0 + (
        (
            score
            - ML_SCAM_THRESHOLD
        )
        / (
            1.0
            - ML_SCAM_THRESHOLD
        )
    ) * 25.0


def calculate_risk(
    signals: set[ScamSignal],
    ml_scam_score: float | None = None,
) -> tuple[float, str]:
    """
    Combine rule-based evidence and ML evidence.

    Returns:
        (risk_score, risk_level)
    """

    rule_score = calculate_rule_score(
        signals
    )

    # Rules-only mode.
    if ml_scam_score is None:
        score = rule_score

    # Hybrid mode.
    else:
        probability = max(
            0.0,
            min(
                float(ml_scam_score),
                1.0,
            ),
        )

        model_risk = model_score_to_risk(
            probability
        )

        # Use whichever detector currently has
        # the stronger evidence.
        score = max(
            rule_score,
            model_risk,
        )

        # ML + meaningful rule evidence.
        if (
            probability >= ML_SCAM_THRESHOLD
            and rule_score >= 20
        ):
            score += 10

        # ML + particularly dangerous requested action.
        if (
            probability >= ML_SCAM_THRESHOLD
            and any(
                signal in HIGH_DANGER_SIGNALS
                for signal in signals
            )
        ):
            score += 10

    score = min(
        float(score),
        100.0,
    )

    if score >= 80:
        level = "CRITICAL"

    elif score >= 50:
        level = "HIGH"

    elif score >= 30:
        level = "CAUTION"

    else:
        level = "SAFE"

    return score, level
