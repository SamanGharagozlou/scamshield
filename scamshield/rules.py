from .signals import ScamSignal


PATTERNS = {
    ScamSignal.BANK_CLAIM: [
        "calling from your bank",
        "calling from the bank",
        "fraud department",
        "bank security",
        "account security",
    ],

    ScamSignal.GOVERNMENT_CLAIM: [
        "calling from the government",
        "tax office",
        "police department",
        "law enforcement",
        "social security administration",
    ],

    ScamSignal.TECH_SUPPORT_CLAIM: [
        "calling from microsoft",
        "technical support",
        "tech support",
        "your computer has a virus",
        "security problem on your computer",
    ],

    ScamSignal.FAMILY_EMERGENCY: [
        "grandma it's me",
        "grandma, it's me",
        "grandpa it's me",
        "grandpa, it's me",
        "i'm using a friend's phone",
        "i am using a friend's phone",
        "had an accident",
    ],

    ScamSignal.OTP_MENTION: [
        "verification code",
        "security code",
        "six digit code",
        "six-digit code",
        "one time password",
        "one-time password",

        # Common paraphrases
        "security message",
        "security notification",
        "message on your phone",
        "appeared on your phone",
        "numbers on your phone",
        "numbers in the message",
    ],

    ScamSignal.OTP_REQUEST: [
        "tell me the code",
        "read me the code",
        "read the code",
        "give me the code",
        "share the code",
        "tell me your verification code",

        # Semantic-style paraphrases
        "tell me the numbers",
        "tell me those numbers",
        "what numbers appeared",
        "what numbers just appeared",
        "numbers that just appeared",
        "read the numbers",
        "read those numbers",
        "read the numbers back",
        "confirm the numbers",
        "say back the numbers",
        "say back the six numbers",
    ],

    ScamSignal.PASSWORD_REQUEST: [
        "tell me your password",
        "give me your password",
        "share your password",
        "confirm your password",
        "provide your password",
    ],

    ScamSignal.PIN_REQUEST: [
        "tell me your pin",
        "give me your pin",
        "share your pin",
        "confirm your pin",
        "provide your pin",
    ],

    ScamSignal.MONEY_TRANSFER_REQUEST: [
        "transfer the money",
        "transfer money",
        "send the money",
        "send money",
        "send me money",
        "make a bank transfer",
        "send funds",
        "make the payment",
        "pay now",
    ],

    ScamSignal.GIFT_CARD_REQUEST: [
        "buy gift cards",
        "purchase gift cards",
        "gift card code",
        "gift card numbers",
        "scratch the back",
    ],

    ScamSignal.REMOTE_ACCESS_REQUEST: [
        "install anydesk",
        "install teamviewer",
        "remote access",
        "give me access to your computer",
        "connect to your computer",
        "control your computer",
        "control your screen",
        "control the screen",
        "open quick assist",
    ],

    ScamSignal.URGENCY: [
        "immediately",
        "right now",
        "urgent",
        "act now",
        "do this now",
        "as soon as possible",
        "before anything else happens",
        "must be done today",
        "payment is required today",
    ],

    ScamSignal.THREAT: [
        "you will be arrested",
        "your account will be closed",
        "your account will be blocked",
        "your account will be suspended",
        "legal action",
        "officers may be sent",
    ],

    ScamSignal.SECRECY: [
        "don't tell anyone",
        "do not tell anyone",
        "keep this confidential",
        "don't tell your family",
        "do not tell your family",
        "keep this between us",
    ],
}


# These phrases indicate the speaker is WARNING the user
# not to perform an unsafe action.
#
# They prevent simple phrase matching from creating
# obvious false positives.
NEGATIVE_PATTERNS = {
    ScamSignal.OTP_REQUEST: [
        "never share",
        "do not share",
        "don't share",
        "never tell me",
        "do not tell me",
        "don't tell me",
        "will never ask",
        "won't ask",
    ],

    ScamSignal.PASSWORD_REQUEST: [
        "never share your password",
        "do not share your password",
        "don't share your password",
        "never tell me your password",
        "do not tell me your password",
        "will never ask for your password",
        "i do not need your password",
    ],

    ScamSignal.PIN_REQUEST: [
        "never share your pin",
        "do not share your pin",
        "don't share your pin",
        "never tell me your pin",
        "do not tell me your pin",
        "will never ask for your pin",
        "i do not need your pin",
    ],

    ScamSignal.MONEY_TRANSFER_REQUEST: [
        "do not send money",
        "don't send money",
        "never send money",
        "do not transfer money",
        "don't transfer money",
        "no payment is required",
        "you do not need to send us any money",
    ],

    ScamSignal.GIFT_CARD_REQUEST: [
        "do not buy gift cards",
        "don't buy gift cards",
        "never buy gift cards",
    ],

    ScamSignal.REMOTE_ACCESS_REQUEST: [
        "do not install anydesk",
        "don't install anydesk",
        "never install anydesk",
        "do not install teamviewer",
        "don't install teamviewer",
        "never install teamviewer",
        "do not give anyone remote access",
        "don't give anyone remote access",
        "will not ask to remotely control",
        "will not ask to control your computer",
    ],
}


def normalize(text: str) -> str:
    return " ".join(
        text.lower().strip().split()
    )


def detect_signals(
    text: str,
) -> list[ScamSignal]:

    normalized = normalize(text)

    detected: list[ScamSignal] = []

    for signal, phrases in PATTERNS.items():

        matched = any(
            phrase in normalized
            for phrase in phrases
        )

        if not matched:
            continue

        negative_phrases = (
            NEGATIVE_PATTERNS.get(
                signal,
                [],
            )
        )

        negated = any(
            phrase in normalized
            for phrase in negative_phrases
        )

        if negated:
            continue

        detected.append(signal)

    return detected
