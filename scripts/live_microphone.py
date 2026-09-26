from pathlib import Path
import argparse
import sys
import time
import uuid

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from scamshield.detector import ScamDetector
from scamshield.models import TranscriptEvent


SAMPLE_RATE = 16000


def print_result(
    text: str,
    result,
    elapsed: float,
) -> None:
    print()
    print("=" * 72)
    print(f"[{elapsed:6.1f}s]")
    print(f'Transcript: "{text}"')
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

    if result.risk_level == "SAFE":
        print()
        print("✅ No strong scam evidence yet.")

    elif result.risk_level == "CAUTION":
        print()
        print("⚠️  CAUTION")
        print(
            "Stay alert and do not share "
            "sensitive information."
        )

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

        action = result.recommended_action

        if action == "DO_NOT_SHARE_CODE":
            print("DO NOT SHARE THE CODE.")

        elif action == "DO_NOT_SHARE_PASSWORD":
            print("DO NOT SHARE YOUR PASSWORD.")

        elif action == "DO_NOT_SHARE_PIN":
            print("DO NOT SHARE YOUR PIN.")

        elif action == "DO_NOT_SEND_MONEY":
            print("DO NOT SEND MONEY.")

        elif action == "DO_NOT_BUY_GIFT_CARDS":
            print("DO NOT BUY GIFT CARDS.")

        elif action == "DO_NOT_ALLOW_REMOTE_ACCESS":
            print("DO NOT ALLOW REMOTE ACCESS.")

        else:
            print("STOP AND VERIFY WHO IS CALLING.")


def record_chunk(
    seconds: float,
) -> np.ndarray:
    frames = int(
        SAMPLE_RATE * seconds
    )

    audio = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    return audio.flatten()


def transcribe_chunk(
    whisper: WhisperModel,
    audio: np.ndarray,
) -> str:
    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    # Ignore very quiet chunks.
    if rms < 0.002:
        return ""

    segments, _ = whisper.transcribe(
        audio,
        language="en",
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False,
    )

    pieces = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            pieces.append(text)

    return " ".join(pieces).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Live microphone scam detection "
            "using Whisper + ScamShield."
        )
    )

    parser.add_argument(
        "--model",
        default="base.en",
        help="Whisper model. Default: base.en",
    )

    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=3.0,
        help=(
            "Length of each microphone chunk. "
            "Default: 3 seconds."
        ),
    )

    args = parser.parse_args()

    conversation_id = (
        "live-"
        + uuid.uuid4().hex[:8]
    )

    print()
    print(
        "SCAMSHIELD LIVE MICROPHONE DEMO"
    )
    print("=" * 72)

    print(
        f"Conversation ID : "
        f"{conversation_id}"
    )

    print(
        f"Chunk duration  : "
        f"{args.chunk_seconds}s"
    )

    print(
        f"Whisper model   : "
        f"{args.model}"
    )

    print(
        f"Sample rate     : "
        f"{SAMPLE_RATE} Hz"
    )

    print()
    print("Loading Whisper...")

    whisper = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
    )

    detector = ScamDetector(
        use_ml=True
    )

    print()
    print("🎙️  Listening...")
    print(
        "Speak normally near the microphone."
    )
    print(
        "Press Ctrl+C to stop."
    )

    start_time = time.monotonic()

    last_result = None
    chunk_number = 0

    try:
        while True:
            chunk_number += 1

            print()
            print(
                f"Listening to chunk "
                f"{chunk_number}..."
            )

            audio = record_chunk(
                args.chunk_seconds
            )

            text = transcribe_chunk(
                whisper,
                audio,
            )

            if not text:
                print(
                    "(No speech detected)"
                )
                continue

            elapsed = (
                time.monotonic()
                - start_time
            )

            event = TranscriptEvent(
                conversation_id=(
                    conversation_id
                ),
                timestamp=elapsed,
                speaker="caller",
                text=text,
            )

            result = detector.process(
                event
            )

            last_result = result

            print_result(
                text,
                result,
                elapsed,
            )

    except KeyboardInterrupt:
        print()
        print()
        print("=" * 72)
        print("LIVE DETECTION STOPPED")
        print("=" * 72)

        if last_result is not None:
            print(
                f"Final risk score : "
                f"{last_result.risk_score:.1f}/100"
            )

            print(
                f"Final risk level : "
                f"{last_result.risk_level}"
            )

            print(
                f"Final category   : "
                f"{last_result.scam_category}"
            )

            print(
                f"Final action     : "
                f"{last_result.recommended_action}"
            )

        else:
            print(
                "No speech was analyzed."
            )


if __name__ == "__main__":
    main()
