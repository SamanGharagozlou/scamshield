from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scamshield.detector import ScamDetector
from scamshield.models import TranscriptEvent


def print_result(text, result):
    print()
    print("=" * 70)
    print(f'Caller: "{text}"')
    print("-" * 70)

    if result.ml_scam_score is not None:
        print(
            f"ML score       : "
            f"{result.ml_scam_score:.3f}"
        )

    print(
        f"Risk score     : "
        f"{result.risk_score:.1f}/100"
    )

    print(
        f"Risk level     : "
        f"{result.risk_level}"
    )

    print(
        f"Signals        : "
        f"{result.signals}"
    )

    print(
        f"Category       : "
        f"{result.scam_category}"
    )

    print(
        f"Action         : "
        f"{result.recommended_action}"
    )

    if result.risk_level == "CAUTION":
        print()
        print("⚠️  CAUTION")

    elif result.risk_level == "HIGH":
        print()
        print("⚠️  POSSIBLE SCAM")
        print(
            "Pause before sharing information "
            "or taking action."
        )

    elif result.risk_level == "CRITICAL":
        print()
        print("🚨 SCAM WARNING")

        if (
            result.recommended_action
            == "DO_NOT_SHARE_CODE"
        ):
            print(
                "DO NOT SHARE THE CODE."
            )

        elif (
            result.recommended_action
            == "DO_NOT_ALLOW_REMOTE_ACCESS"
        ):
            print(
                "DO NOT ALLOW REMOTE ACCESS."
            )

        elif (
            result.recommended_action
            == "DO_NOT_SEND_MONEY"
        ):
            print(
                "DO NOT SEND MONEY."
            )

        elif (
            result.recommended_action
            == "DO_NOT_SHARE_PASSWORD"
        ):
            print(
                "DO NOT SHARE YOUR PASSWORD."
            )


def main():
    detector = ScamDetector(
        use_ml=True
    )

    conversation_id = (
        "realtime-bank-demo"
    )

    conversation = [
        (
            3,
            "Hello, I'm calling from your bank."
        ),
        (
            10,
            "We've noticed some unusual "
            "activity on your account."
        ),
        (
            17,
            "We need to secure the account "
            "before anything else happens."
        ),
        (
            24,
            "A security message should have "
            "appeared on your phone."
        ),
        (
            31,
            "Can you tell me the numbers "
            "that just appeared?"
        ),
    ]

    print()
    print(
        "SCAMSHIELD REAL-TIME DETECTION DEMO"
    )
    print("=" * 70)

    for timestamp, text in conversation:

        result = detector.process(
            TranscriptEvent(
                conversation_id=conversation_id,
                timestamp=float(timestamp),
                speaker="caller",
                text=text,
            )
        )

        print_result(
            text,
            result,
        )

        time.sleep(1)


if __name__ == "__main__":
    main()
