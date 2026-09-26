import json
import sys
import time
from pathlib import Path

# Make the project root importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scamshield.detector import ScamDetector
from scamshield.models import TranscriptEvent


def load_scenario(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_scenario(path: str, delay: float = 1.0) -> None:
    scenario = load_scenario(path)

    detector = ScamDetector()

    print()
    print("=" * 70)
    print(f"ScamShield Demo: {scenario['id']}")
    print(f"Expected label: {scenario['label']}")
    print("=" * 70)

    for turn in scenario["turns"]:
        event = TranscriptEvent(
            conversation_id=scenario["id"],
            timestamp=float(turn["timestamp"]),
            speaker=turn["speaker"],
            text=turn["text"],
        )

        result = detector.process(event)

        print()
        print(f"[{turn['timestamp']:>5}s] {turn['speaker'].upper()}")
        print(turn["text"])
        print()
        print(f"Risk score : {result.risk_score:.0f}/100")
        print(f"Risk level : {result.risk_level}")

        if result.signals:
            print("Signals    : " + ", ".join(result.signals))
        else:
            print("Signals    : None")

        if result.scam_category:
            print(f"Category   : {result.scam_category}")

        if result.recommended_action:
            print(f"ACTION     : {result.recommended_action}")

        if result.risk_level == "CRITICAL":
            print()
            print("!" * 70)
            print("SCAMSHIELD WARNING")
            print("STOP - THIS MAY BE A SCAM")
            if result.recommended_action:
                print(result.recommended_action)
            print("!" * 70)

        print("-" * 70)

        time.sleep(delay)


if __name__ == "__main__":
    default_path = "scenarios/bank_otp_scam.json"

    scenario_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else default_path
    )

    if not Path(scenario_path).exists():
        raise SystemExit(
            f"Scenario not found: {scenario_path}"
        )

    run_scenario(scenario_path)
