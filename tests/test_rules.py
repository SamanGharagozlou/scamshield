from scamshield.rules import detect_signals
from scamshield.signals import ScamSignal


def test_detects_bank_claim():
    signals = detect_signals(
        "Hello, I'm calling from your bank."
    )

    assert ScamSignal.BANK_CLAIM in signals


def test_detects_otp_request():
    signals = detect_signals(
        "Please read the code to me."
    )

    assert ScamSignal.OTP_REQUEST in signals


def test_detects_urgency():
    signals = detect_signals(
        "We need to fix this immediately."
    )

    assert ScamSignal.URGENCY in signals


def test_detects_remote_access_request():
    signals = detect_signals(
        "Please install AnyDesk so I can access your computer."
    )

    assert ScamSignal.REMOTE_ACCESS_REQUEST in signals


def test_detects_secrecy():
    signals = detect_signals(
        "Don't tell your family about this."
    )

    assert ScamSignal.SECRECY in signals


def test_safe_sentence_returns_no_signal():
    signals = detect_signals(
        "Hello, how are you doing today?"
    )

    assert signals == []
