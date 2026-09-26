from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from faster_whisper import WhisperModel

from scamshield.detector import ScamDetector
from scamshield.models import TranscriptEvent


def print_result(segment, result):
    print()
    print("=" * 72)

    print(
        f"[{segment.start:6.2f}s → {segment.end:6.2f}s]"
    )

    print(
        f'Transcript: "{segment.text.strip()}"'
    )

    print("-" * 72)

    if result.ml_scam_score is not None:
        print(
            f"ML score           : "
            f"{result.ml_scam_score:.3f}"
        )

    print(
        f"Risk score         : "
        f"{result.risk_score:.1f}/100"
    )

    print(
        f"Risk level         : "
        f"{result.risk_level}"
    )

    print(
        f"Signals            : "
        f"{result.signals}"
    )

    print(
        f"Scam category      : "
        f"{result.scam_category}"
    )

    print(
        f"Recommended action : "
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

        if result.recommended_action == "DO_NOT_SHARE_CODE":
            print("DO NOT SHARE THE CODE.")

        elif (
            result.recommended_action
            == "DO_NOT_SHARE_PASSWORD"
        ):
            print(
                "DO NOT SHARE YOUR PASSWORD."
            )

        elif (
            result.recommended_action
            == "DO_NOT_SHARE_PIN"
        ):
            print(
                "DO NOT SHARE YOUR PIN."
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
            == "DO_NOT_BUY_GIFT_CARDS"
        ):
            print(
                "DO NOT BUY GIFT CARDS."
            )

        elif (
            result.recommended_action
            == "DO_NOT_ALLOW_REMOTE_ACCESS"
        ):
            print(
                "DO NOT ALLOW REMOTE ACCESS."
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an audio call with Whisper "
            "and run each segment through ScamShield."
        )
    )

    parser.add_argument(
        "audio",
        help="Path to the audio file.",
    )

    parser.add_argument(
        "--model",
        default="base.en",
        help="Whisper model. Default: base.en",
    )

    parser.add_argument(
        "--conversation-id",
        default="audio-demo",
        help="Conversation ID used by ScamShield.",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print()
    print(
        "SCAMSHIELD AUDIO → AI DETECTION"
    )
    print("=" * 72)

    print(
        f"Audio: {audio_path}"
    )

    print(
        f"Whisper model: {args.model}"
    )

    print()
    print(
        "Loading Whisper..."
    )

    whisper = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
    )

    detector = ScamDetector(
        use_ml=True
    )

    print(
        "Transcribing and analyzing..."
    )

    segments, info = whisper.transcribe(
        str(audio_path),
        language="en",
        vad_filter=True,
    )

    print(
        f"Detected language: {info.language}"
    )

    segment_count = 0
    last_result = None

    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue

        segment_count += 1

        event = TranscriptEvent(
            conversation_id=(
                args.conversation_id
            ),
            timestamp=float(
                segment.end
            ),
            speaker="caller",
            text=text,
        )

        result = detector.process(
            event
        )

        last_result = result

        print_result(
            segment,
            result,
        )

    print()
    print("=" * 72)
    print("FINAL CALL ASSESSMENT")
    print("=" * 72)

    if last_result is None:
        print(
            "No speech segments were detected."
        )

        return

    print(
        f"Segments analyzed : "
        f"{segment_count}"
    )

    print(
        f"Final risk score  : "
        f"{last_result.risk_score:.1f}/100"
    )

    print(
        f"Final risk level  : "
        f"{last_result.risk_level}"
    )

    print(
        f"Category          : "
        f"{last_result.scam_category}"
    )

    print(
        f"Action            : "
        f"{last_result.recommended_action}"
    )


if __name__ == "__main__":
    main()
