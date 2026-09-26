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


TEST_PATH = Path("data/processed/test.jsonl")


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def print_results(name, y_true, y_pred):
    result = metrics(y_true, y_pred)

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    for key, value in result.items():
        print(f"{key:10s}: {value:.4f}")

    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def main():
    df = pd.read_json(
        TEST_PATH,
        lines=True,
    )

    print(f"Loaded {len(df)} test conversations")

    y_true = df["label"].tolist()

    # ------------------------------------------------
    # 1. RULES ONLY
    # ------------------------------------------------

    rule_predictions = []

    for text in df["text"]:
        signals = set(detect_signals(text))

        score, level = calculate_risk(
            signals,
            ml_scam_score=None,
        )

        # Baseline binary decision:
        # HIGH or CRITICAL = scam
        prediction = 1 if score >= 50 else 0

        rule_predictions.append(prediction)

    # ------------------------------------------------
    # 2. ML ONLY
    # ------------------------------------------------

    classifier = SemanticScamClassifier()

    ml_scores = [
        classifier.predict_score(text)
        for text in df["text"]
    ]

    ml_predictions = [
        1 if score >= 0.50 else 0
        for score in ml_scores
    ]

    # ------------------------------------------------
    # 3. HYBRID
    # ------------------------------------------------

    hybrid_predictions = []
    hybrid_scores = []

    for i, text in enumerate(df["text"]):

        detector = ScamDetector(
            use_ml=True
        )

        result = detector.process(
            TranscriptEvent(
                conversation_id=f"eval-{i}",
                timestamp=0.0,
                speaker="caller",
                text=text,
            )
        )

        hybrid_scores.append(
            result.risk_score
        )

        prediction = (
            1
            if result.risk_score >= 50
            else 0
        )

        hybrid_predictions.append(
            prediction
        )

    # ------------------------------------------------
    # RESULTS
    # ------------------------------------------------

    print_results(
        "RULES ONLY",
        y_true,
        rule_predictions,
    )

    print_results(
        "ML ONLY",
        y_true,
        ml_predictions,
    )

    print_results(
        "HYBRID",
        y_true,
        hybrid_predictions,
    )

    # ------------------------------------------------
    # ERROR ANALYSIS
    # ------------------------------------------------

    output = df[
        [
            "id",
            "text",
            "label",
            "source_type",
        ]
    ].copy()

    output["ml_score"] = ml_scores
    output["hybrid_score"] = hybrid_scores

    output["rules_prediction"] = (
        rule_predictions
    )

    output["ml_prediction"] = (
        ml_predictions
    )

    output["hybrid_prediction"] = (
        hybrid_predictions
    )

    errors = output[
        output["hybrid_prediction"]
        != output["label"]
    ]

    error_path = Path(
        "data/processed/hybrid_errors.csv"
    )

    errors.to_csv(
        error_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("HYBRID ERRORS")
    print("=" * 60)

    print(
        f"{len(errors)} incorrect out of "
        f"{len(output)} conversations"
    )

    print(
        f"Saved error analysis → {error_path}"
    )

    print()
    print("FIRST 10 ERRORS")

    for _, row in errors.head(10).iterrows():

        print()
        print("-" * 60)

        print(
            f"ID: {row['id']}"
        )

        print(
            f"Type: {row['source_type']}"
        )

        print(
            f"Actual: {row['label']}"
        )

        print(
            f"ML score: "
            f"{row['ml_score']:.3f}"
        )

        print(
            f"Hybrid score: "
            f"{row['hybrid_score']:.1f}"
        )

        print(
            row["text"][:500]
        )


if __name__ == "__main__":
    main()
