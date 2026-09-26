from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from scamshield.classifier import SemanticScamClassifier


PATH = Path(
    "data/external/menaattia_calibration.jsonl"
)


def main():
    df = pd.read_json(
        PATH,
        lines=True,
    )

    classifier = SemanticScamClassifier()

    scores = [
        classifier.predict_score(str(text))
        for text in df["dialogue"]
    ]

    y_true = df["label"].astype(int).tolist()

    results = []

    for threshold_int in range(25, 61):

        threshold = threshold_int / 100

        predictions = [
            int(score >= threshold)
            for score in scores
        ]

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        # F2 weights recall more heavily than precision.
        beta = 2

        if precision + recall == 0:
            f2 = 0
        else:
            f2 = (
                (1 + beta**2)
                * precision
                * recall
                / (
                    beta**2 * precision
                    + recall
                )
            )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
        })

    result_df = pd.DataFrame(results)

    print()
    print("ALL THRESHOLDS")
    print(
        result_df.to_string(
            index=False,
            formatters={
                "threshold": lambda x: f"{x:.2f}",
                "precision": lambda x: f"{x:.3f}",
                "recall": lambda x: f"{x:.3f}",
                "f1": lambda x: f"{x:.3f}",
                "f2": lambda x: f"{x:.3f}",
            }
        )
    )

    # For ScamShield we care strongly about recall,
    # but don't want terrible precision.
    candidates = result_df[
        result_df["precision"] >= 0.85
    ]

    if len(candidates):
        best = candidates.loc[
            candidates["f2"].idxmax()
        ]
    else:
        best = result_df.loc[
            result_df["f2"].idxmax()
        ]

    print()
    print("=" * 50)
    print("RECOMMENDED CALIBRATION THRESHOLD")
    print("=" * 50)

    print(
        f"Threshold : {best['threshold']:.2f}"
    )

    print(
        f"Precision : {best['precision']:.3f}"
    )

    print(
        f"Recall    : {best['recall']:.3f}"
    )

    print(
        f"F1        : {best['f1']:.3f}"
    )

    print(
        f"F2        : {best['f2']:.3f}"
    )


if __name__ == "__main__":
    main()
