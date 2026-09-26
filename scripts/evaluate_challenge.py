from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from scamshield.classifier import SemanticScamClassifier
from scamshield.detector import ScamDetector
from scamshield.models import TranscriptEvent
from scamshield.rules import detect_signals
from scamshield.risk import calculate_risk


PATH = Path("data/benchmark/scamshield_challenge.jsonl")


def show_metrics(name, y_true, y_pred):
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1        : {f1_score(y_true, y_pred, zero_division=0):.4f}")

    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def main():
    df = pd.read_json(PATH, lines=True)

    classifier = SemanticScamClassifier()

    y_true = df["label"].tolist()

    rule_predictions = []
    ml_predictions = []
    hybrid_predictions = []

    rows = []

    for i, row in df.iterrows():
        text = row["text"]

        # Rules
        signals = set(detect_signals(text))
        rule_score, _ = calculate_risk(
            signals,
            ml_scam_score=None,
        )
        rule_pred = int(rule_score >= 50)

        # ML
        ml_score = classifier.predict_score(text)
        ml_pred = int(ml_score >= 0.50)

        # Hybrid
        detector = ScamDetector(use_ml=True)

        result = detector.process(
            TranscriptEvent(
                conversation_id=f"challenge-{i}",
                timestamp=0.0,
                speaker="caller",
                text=text,
            )
        )

        hybrid_pred = int(result.risk_score >= 50)

        rule_predictions.append(rule_pred)
        ml_predictions.append(ml_pred)
        hybrid_predictions.append(hybrid_pred)

        rows.append({
            "id": row["id"],
            "category": row["category"],
            "actual": row["label"],
            "rules_score": rule_score,
            "ml_score": ml_score,
            "hybrid_score": result.risk_score,
            "rules_pred": rule_pred,
            "ml_pred": ml_pred,
            "hybrid_pred": hybrid_pred,
        })

    show_metrics(
        "RULES ONLY — CHALLENGE SET",
        y_true,
        rule_predictions,
    )

    show_metrics(
        "ML ONLY — CHALLENGE SET",
        y_true,
        ml_predictions,
    )

    show_metrics(
        "HYBRID — CHALLENGE SET",
        y_true,
        hybrid_predictions,
    )

    results = pd.DataFrame(rows)

    print()
    print("=" * 60)
    print("INDIVIDUAL RESULTS")
    print("=" * 60)

    print(
        results.to_string(
            index=False,
            formatters={
                "ml_score": lambda x: f"{x:.3f}",
                "hybrid_score": lambda x: f"{x:.1f}",
            },
        )
    )

    errors = results[
        results["hybrid_pred"] != results["actual"]
    ]

    print()
    print("=" * 60)
    print(f"HYBRID ERRORS: {len(errors)}")
    print("=" * 60)

    if len(errors):
        print(errors.to_string(index=False))
    else:
        print("No hybrid errors.")


if __name__ == "__main__":
    main()
