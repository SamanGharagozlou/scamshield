from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline


DATA_DIR = Path("data/processed")

AUGMENTATION_PATH = Path(
    "data/augmentation/scamshield_train.jsonl"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def extract_caller_text(text: str) -> str:
    """
    Extract only speech belonging to 'caller:'.

    Example:

    caller: Hello.
    receiver: Who is this?
    caller: I'm calling from your bank.

    becomes:

    Hello. I'm calling from your bank.

    If there are no caller/receiver markers,
    keep the text unchanged. This is useful for
    our curated ScamShield examples.
    """

    text = str(text).strip()

    if not text:
        return ""

    has_speaker_markers = re.search(
        r"\b(?:caller|receiver)\s*:",
        text,
        flags=re.IGNORECASE,
    )

    if not has_speaker_markers:
        return text

    parts = re.split(
        r"\b(caller|receiver)\s*:\s*",
        text,
        flags=re.IGNORECASE,
    )

    caller_parts = []

    # After splitting:
    #
    # [
    #   prefix,
    #   caller,
    #   speech,
    #   receiver,
    #   speech,
    #   caller,
    #   speech,
    #   ...
    # ]
    for index in range(
        1,
        len(parts) - 1,
        2,
    ):
        role = (
            parts[index]
            .strip()
            .lower()
        )

        speech = (
            parts[index + 1]
            .strip()
        )

        if (
            role == "caller"
            and speech
        ):
            caller_parts.append(
                speech
            )

    return " ".join(
        caller_parts
    ).strip()


def load_split(
    filename: str,
) -> pd.DataFrame:

    return pd.read_json(
        DATA_DIR / filename,
        lines=True,
    )


def prepare_split(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["caller_text"] = (
        df["text"]
        .astype(str)
        .apply(extract_caller_text)
    )

    # Remove unusable empty examples.
    df = df[
        df["caller_text"].str.len() > 0
    ].copy()

    return df.reset_index(
        drop=True
    )


def evaluate(
    name: str,
    df: pd.DataFrame,
    model,
) -> None:

    y_true = (
        df["label"]
        .astype(int)
    )

    predictions = model.predict(
        df["caller_text"]
    )

    probabilities = model.predict_proba(
        df["caller_text"]
    )[:, 1]

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Accuracy  : "
        f"{accuracy_score(y_true, predictions):.4f}"
    )

    print(
        f"Precision : "
        f"{precision_score(y_true, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall    : "
        f"{recall_score(y_true, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1        : "
        f"{f1_score(y_true, predictions, zero_division=0):.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{roc_auc_score(y_true, probabilities):.4f}"
    )

    print()
    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions,
        )
    )

    print()
    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "SAFE",
                "SCAM",
            ],
        )
    )


def main():
    print(
        "Loading ScamShield datasets..."
    )

    train = prepare_split(
        load_split("train.jsonl")
    )

    validation = prepare_split(
        load_split("validation.jsonl")
    )

    test = prepare_split(
        load_split("test.jsonl")
    )

    print()
    print(
        f"Base training rows : {len(train)}"
    )

    print(
        f"Validation rows    : {len(validation)}"
    )

    print(
        f"Test rows          : {len(test)}"
    )

    # ------------------------------------------
    # ScamShield curated augmentation examples
    # ------------------------------------------

    training_data = train[
        [
            "caller_text",
            "label",
        ]
    ].copy()

    if AUGMENTATION_PATH.exists():

        augmentation = pd.read_json(
            AUGMENTATION_PATH,
            lines=True,
        )

        augmentation[
            "caller_text"
        ] = (
            augmentation["text"]
            .astype(str)
            .apply(extract_caller_text)
        )

        augmentation = augmentation[
            [
                "caller_text",
                "label",
            ]
        ]

        # Same weighting used by our existing model.
        augmentation = pd.concat(
            [augmentation] * 5,
            ignore_index=True,
        )

        training_data = pd.concat(
            [
                training_data,
                augmentation,
            ],
            ignore_index=True,
        )

    print(
        f"Final training rows: "
        f"{len(training_data)}"
    )

    # ------------------------------------------
    # Train caller-only classifier
    # ------------------------------------------

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print()
    print(
        "Training caller-only classifier..."
    )

    model.fit(
        training_data["caller_text"],
        training_data["label"],
    )

    print(
        "Training complete."
    )

    # ------------------------------------------
    # Evaluate on original dataset splits
    # ------------------------------------------

    evaluate(
        "CALLER-ONLY VALIDATION",
        validation,
        model,
    )

    evaluate(
        "CALLER-ONLY TEST",
        test,
        model,
    )

    # ------------------------------------------
    # Save WITHOUT replacing current model
    # ------------------------------------------

    model_path = (
        MODEL_DIR
        / "scam_classifier_caller.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print()
    print(
        f"Saved caller-only model → "
        f"{model_path}"
    )

    # ------------------------------------------
    # Verify caller extraction
    # ------------------------------------------

    original_df = load_split(
        "train.jsonl"
    )

    original = str(
        original_df.iloc[0]["text"]
    )

    caller_only = (
        extract_caller_text(
            original
        )
    )

    print()
    print("=" * 60)
    print("EXTRACTION CHECK")
    print("=" * 60)

    print()
    print("ORIGINAL DIALOGUE:")
    print(
        original[:1500]
    )

    print()
    print("CALLER ONLY:")
    print(
        caller_only[:1500]
    )


if __name__ == "__main__":
    main()
