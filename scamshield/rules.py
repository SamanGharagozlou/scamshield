from .signals import ScamSignal


PATTERNS = {
    ScamSignal.BANK_CLAIM: [
        "calling from your bank",
        "fraud department",
        "bank security",
        "calling from the bank",
    ],

    ScamSignal.GOVERNMENT_CLAIM: [
        "calling from the government",
        "tax office",
        "police department",
        "law enforcement",
    ],

    ScamSignal.TECH_SUPPORT_CLAIM: [
        "calling from microsoft",
        "technical support",
        "tech support",
        "your computer has a virus",
    ],

    ScamSignal.OTP_MENTION: [
        "verification code",
        "security code",
        "six digit code",
        "six-digit code",
        "one time password",
        "one-time password",
    ],

    ScamSignal.OTP_REQUEST: [
        "tell me the code",
        "read me the code",
        "read the code",
        "give me the code",
        "share the code",
        "tell me your verification code",
    ],

    ScamSignal.PASSWORD_REQUEST: [
        "tell me your password",
        "give me your password",
        "share your password",
    ],

    ScamSignal.PIN_REQUEST: [
        "tell me your pin",
        "give me your pin",
        "share your pin",
    ],

    ScamSignal.MONEY_TRANSFER_REQUEST: [
        "transfer the money",
        "send the money",
        "make a bank transfer",
        "send funds",
    ],

    ScamSignal.GIFT_CARD_REQUEST: [
        "buy gift cards",
        "purchase gift cards",
        "gift card code",
    ],

    ScamSignal.REMOTE_ACCESS_REQUEST: [
        "install anydesk",
        "install teamviewer",
        "remote access",
        "give me access to your computer",
    ],

    ScamSignal.URGENCY: [
        "immediately",
        "right now",
        "urgent",
        "act now",
        "do this now",
        "as soon as possible",
    ],

    ScamSignal.THREAT: [
        "you will be arrested",
        "your account will be closed",
        "your account will be blocked",
        "legal action",
    ],

    ScamSignal.SECRECY: [
        "don't tell anyone",
        "do not tell anyone",
        "keep this confidential",
        "don't tell your family",
    ],
}


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def detect_signals(text: str) -> list[ScamSignal]:
    normalized = normalize(text)

    detected: list[ScamSignal] = []

    for signal, phrases in PATTERNS.items():
        if any(phrase in normalized for phrase in phrases):
            detected.append(signal)

    return detected
