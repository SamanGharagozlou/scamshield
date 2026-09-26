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
    roc_auc_score,
)

from scamshield.classifier import SemanticScamClassifier
from scamshield.detector import ScamDetector
from scamshield.models import TranscriptEvent
from scamshield.rules import detect_signals
from scamshield.risk import calculate_risk


PATH = Path(
    "data/external/menaattia_test.jsonl"
)


def print_metrics(
    name,
    y_true,
    y_pred,
    scores=None,
):
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Accuracy  : "
        f"{accuracy_score(y_true, y_pred):.4f}"
    )

    print(
        f"Precision : "
        f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"Recall    : "
        f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"F1        : "
        f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
    )

    if scores is not None:
        print(
            f"ROC-AUC   : "
            f"{roc_auc_score(y_true, scores):.4f}"
        )

    print()
    print("Confusion matrix:")
    print(confusion_matrix(
        y_true,
        y_pred,
    ))


def main():

    df = pd.read_json(
        PATH,
        lines=True,
    )

    print(
        f"Loaded {len(df)} external test conversations"
    )

    print(
        "Label counts:"
    )
    print(
        df["label"].value_counts()
    )

    y_true = df[
        "label"
    ].astype(int).tolist()

    classifier = SemanticScamClassifier()

    rules_predictions = []

    ml_scores = []
    ml_predictions = []

    hybrid_scores = []
    hybrid_predictions = []

    error_rows = []

    for i, row in df.iterrows():

        text = str(
            row["dialogue"]
        )

        # -------------------------
        # RULES ONLY
        # -------------------------

        signals = set(
            detect_signals(text)
        )

        rule_score, _ = calculate_risk(
            signals,
            ml_scam_score=None,
        )

        rule_pred = int(
            rule_score >= 50
        )

        rules_predictions.append(
            rule_pred
        )

        # -------------------------
        # ML ONLY
        # -------------------------

        ml_score = (
            classifier.predict_score(
                text
            )
        )

        ml_pred = int(
            ml_score >= 0.50
        )

        ml_scores.append(
            ml_score
        )

        ml_predictions.append(
            ml_pred
        )

        # -------------------------
        # HYBRID
        # -------------------------

        detector = ScamDetector(
            use_ml=True
        )

        result = detector.process(
            TranscriptEvent(
                conversation_id=f"external-{i}",
                timestamp=0.0,
                speaker="caller",
                text=text,
            )
        )

        hybrid_pred = int(
            result.risk_score >= 50
        )

        hybrid_scores.append(
            result.risk_score
        )

        hybrid_predictions.append(
            hybrid_pred
        )

        if hybrid_pred != int(
            row["label"]
        ):
            error_rows.append({
                "index": i,
                "actual": int(
                    row["label"]
                ),
                "ml_score": ml_score,
                "hybrid_score": (
                    result.risk_score
                ),
                "text": text,
            })

    print_metrics(
        "RULES ONLY — EXTERNAL",
        y_true,
        rules_predictions,
    )

    print_metrics(
        "ML ONLY — EXTERNAL",
        y_true,
        ml_predictions,
        ml_scores,
    )

    print_metrics(
        "HYBRID — EXTERNAL",
        y_true,
        hybrid_predictions,
        [
            score / 100
            for score in hybrid_scores
        ],
    )

    errors = pd.DataFrame(
        error_rows
    )

    error_path = Path(
        "data/external/external_errors.csv"
    )

    errors.to_csv(
        error_path,
        index=False,
    )

    print()
    print("=" * 60)
    print("EXTERNAL HYBRID ERRORS")
    print("=" * 60)

    print(
        f"{len(errors)} errors "
        f"out of {len(df)}"
    )

    print(
        f"Saved → {error_path}"
    )

    if len(errors):
        print()
        print("FIRST 15 ERRORS")

        for _, row in (
            errors.head(15).iterrows()
        ):
            print()
            print("-" * 60)

            print(
                f"Actual: {row['actual']}"
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
                row["text"][:600]
            )


if __name__ == "__main__":
    main()
