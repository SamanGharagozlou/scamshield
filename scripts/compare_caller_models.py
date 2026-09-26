from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


EXTERNAL_TEST = Path(
    "data/external/menaattia_test.jsonl"
)

FULL_MODEL_PATH = Path(
    "models/scam_classifier.joblib"
)

CALLER_MODEL_PATH = Path(
    "models/scam_classifier_caller.joblib"
)

THRESHOLD = 0.40


def extract_caller_text(text: str) -> str:
    text = str(text).strip()

    if not text:
        return ""

    if not re.search(
        r"\b(?:caller|receiver)\s*:",
        text,
        flags=re.IGNORECASE,
    ):
        return text

    parts = re.split(
        r"\b(caller|receiver)\s*:\s*",
        text,
        flags=re.IGNORECASE,
    )

    caller_parts = []

    for index in range(
        1,
        len(parts) - 1,
        2,
    ):
        role = parts[index].strip().lower()
        speech = parts[index + 1].strip()

        if role == "caller" and speech:
            caller_parts.append(speech)

    return " ".join(caller_parts).strip()


def evaluate(
    name,
    model,
    texts,
    y_true,
):
    scores = model.predict_proba(
        texts
    )[:, 1]

    predictions = [
        int(score >= THRESHOLD)
        for score in scores
    ]

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

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

    auc = roc_auc_score(
        y_true,
        scores,
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Threshold : {THRESHOLD:.2f}"
    )
    print(
        f"Accuracy  : {accuracy:.4f}"
    )
    print(
        f"Precision : {precision:.4f}"
    )
    print(
        f"Recall    : {recall:.4f}"
    )
    print(
        f"F1        : {f1:.4f}"
    )
    print(
        f"ROC-AUC   : {auc:.4f}"
    )

    print()
    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions,
        )
    )

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "scores": scores,
        "predictions": predictions,
    }


def main():
    print(
        "Loading untouched external test set..."
    )

    df = pd.read_json(
        EXTERNAL_TEST,
        lines=True,
    )

    print(
        f"Rows: {len(df)}"
    )

    print()
    print("Labels:")
    print(
        df["label"].value_counts()
    )

    y_true = (
        df["label"]
        .astype(int)
        .tolist()
    )

    full_text = (
        df["dialogue"]
        .astype(str)
        .tolist()
    )

    caller_text = [
        extract_caller_text(text)
        for text in full_text
    ]

    empty_count = sum(
        not text.strip()
        for text in caller_text
    )

    print()
    print(
        f"Empty caller-only rows: "
        f"{empty_count}"
    )

    if empty_count:
        raise RuntimeError(
            "Caller extraction produced "
            "empty examples."
        )

    print()
    print("Loading models...")

    full_model = joblib.load(
        FULL_MODEL_PATH
    )

    caller_model = joblib.load(
        CALLER_MODEL_PATH
    )

    # ------------------------------------------------
    # A. Existing model on original full dialogue
    # ------------------------------------------------

    result_a = evaluate(
        (
            "A — CURRENT MODEL "
            "+ FULL DIALOGUE"
        ),
        full_model,
        full_text,
        y_true,
    )

    # ------------------------------------------------
    # B. Existing model on caller-only dialogue
    # ------------------------------------------------

    result_b = evaluate(
        (
            "B — CURRENT MODEL "
            "+ CALLER ONLY"
        ),
        full_model,
        caller_text,
        y_true,
    )

    # ------------------------------------------------
    # C. Caller-trained model on caller-only dialogue
    # ------------------------------------------------

    result_c = evaluate(
        (
            "C — CALLER-ONLY MODEL "
            "+ CALLER ONLY"
        ),
        caller_model,
        caller_text,
        y_true,
    )

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(
        f"{'Model':42s} "
        f"{'Precision':>9s} "
        f"{'Recall':>9s} "
        f"{'F1':>9s} "
        f"{'AUC':>9s}"
    )

    print("-" * 82)

    for result in [
        result_a,
        result_b,
        result_c,
    ]:
        print(
            f"{result['name'][:42]:42s} "
            f"{result['precision']:9.4f} "
            f"{result['recall']:9.4f} "
            f"{result['f1']:9.4f} "
            f"{result['auc']:9.4f}"
        )

    # ------------------------------------------------
    # Show caller extraction examples
    # ------------------------------------------------

    print()
    print("=" * 70)
    print("EXTERNAL CALLER EXTRACTION EXAMPLES")
    print("=" * 70)

    for i in range(3):
        print()
        print(f"EXAMPLE {i + 1}")
        print("-" * 70)

        print("FULL:")
        print(
            full_text[i][:700]
        )

        print()
        print("CALLER ONLY:")
        print(
            caller_text[i][:700]
        )


if __name__ == "__main__":
    main()
