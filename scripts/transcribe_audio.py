from pathlib import Path
import argparse

from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a ScamShield demo call."
    )

    parser.add_argument(
        "audio",
        help="Path to WAV, M4A, MP3, or other supported audio file.",
    )

    parser.add_argument(
        "--model",
        default="base.en",
        help="Whisper model name. Default: base.en",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print()
    print("SCAMSHIELD SPEECH-TO-TEXT TEST")
    print("=" * 70)

    print(f"Audio: {audio_path}")
    print(f"Model: {args.model}")

    print()
    print("Loading Whisper model...")

    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
    )

    print("Transcribing...")
    print()

    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        vad_filter=True,
    )

    print(
        f"Detected language: "
        f"{info.language}"
    )

    print()
    print("=" * 70)
    print("TRANSCRIPT")
    print("=" * 70)

    count = 0

    for segment in segments:
        count += 1

        text = segment.text.strip()

        print(
            f"[{segment.start:6.2f}s "
            f"→ {segment.end:6.2f}s] "
            f"{text}"
        )

    print()
    print("=" * 70)
    print(f"Segments: {count}")


if __name__ == "__main__":
    main()
