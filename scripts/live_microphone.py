from pathlib import Path
import sys
import time
import uuid

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from scamshield.alerts import critical_alerts
from scamshield.detector import ScamDetector
from scamshield.models import TranscriptEvent


SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
MIN_RMS = 0.002

WHISPER_MODEL = "base.en"


def print_result(
    transcript: str,
    result,
) -> None:

    print()
    print("=" * 70)
    print(f'TRANSCRIPT: "{transcript}"')
    print("-" * 70)

    if result.ml_scam_score is not None:
        print(
            f"ML score       : "
            f"{result.ml_scam_score:.3f}"
        )

    print(
        f"Risk score     : "
        f"{result.risk_score:.1f}"
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
        f"Scam category  : "
        f"{result.scam_category}"
    )

    print(
        f"Action         : "
        f"{result.recommended_action}"
    )

    if result.risk_level == "CAUTION":
        print()
        print("⚠️  CAUTION — suspicious conversation.")

    elif result.risk_level == "HIGH":
        print()
        print("⚠️⚠️  HIGH RISK — do not share sensitive information.")

    elif result.risk_level == "CRITICAL":
        print()
        print("🚨🚨🚨 CRITICAL SCAM WARNING 🚨🚨🚨")

        if result.recommended_action == "DO_NOT_SHARE_CODE":
            print(
                "⛔ DO NOT SHARE THE VERIFICATION CODE."
            )

        elif result.recommended_action:
            print(
                f"⛔ {result.recommended_action}"
            )

        else:
            print(
                "⛔ STOP. Do not follow the caller's instructions."
            )

    print("=" * 70)


def main():

    print()
    print("Loading Whisper...")
    print(
        "The first run may take a little longer "
        "while the model loads."
    )

    whisper = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
    )

    detector = ScamDetector()

    conversation_id = (
        f"live-{uuid.uuid4()}"
    )

    print()
    print("=" * 70)
    print("SCAMSHIELD LIVE MICROPHONE")
    print("=" * 70)

    print(
        f"Conversation ID: "
        f"{conversation_id}"
    )

    print()
    print(
        "Speak into the Mac microphone."
    )

    print(
        f"Listening in {CHUNK_SECONDS}-second chunks."
    )

    print(
        "Press Ctrl+C to stop."
    )

    print()

    try:

        while True:

            print(
                "🎙️  Listening...",
                flush=True,
            )

            audio = sd.rec(
                int(
                    CHUNK_SECONDS
                    * SAMPLE_RATE
                ),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )

            sd.wait()

            audio = np.squeeze(
                audio
            )

            rms = float(
                np.sqrt(
                    np.mean(
                        np.square(audio)
                    )
                )
            )

            if rms < MIN_RMS:
                print(
                    "   No clear speech detected."
                )
                continue

            segments, _ = whisper.transcribe(
                audio,
                language="en",
                vad_filter=True,
                beam_size=1,
                condition_on_previous_text=False,
            )

            transcript_parts = [
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ]

            transcript = " ".join(
                transcript_parts
            ).strip()

            if not transcript:
                print(
                    "   No transcript detected."
                )
                continue

            event = TranscriptEvent(
                conversation_id=conversation_id,
                timestamp=time.time(),
                speaker="caller",
                text=transcript,
            )

            result = detector.process(
                event
            )

            print_result(
                transcript,
                result,
            )

            # --------------------------------------
            # n8n trusted-contact escalation
            # --------------------------------------

            critical_alerts.send_if_needed(
                risk_event=result,
                latest_text=transcript,
            )

    except KeyboardInterrupt:

        print()
        print()
        print(
            "ScamShield live microphone stopped."
        )


if __name__ == "__main__":
    main()
