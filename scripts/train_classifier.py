from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename

    return pd.read_json(
        path,
        lines=True,
    )


def main():
    print("Loading ScamShield dataset...")

    train = load_jsonl("train.jsonl")
    augmentation = pd.read_json(
    "data/augmentation/scamshield_train.jsonl",
    lines=True,
    )

    augmentation = augmentation[
    ["text", "label"]
    ]

    # Give our difficult ScamShield-specific examples
    # more influence than one ordinary synthetic row.
    augmentation = pd.concat(
    [augmentation] * 5,
    ignore_index=True,
)

    train = pd.concat(
    [
        train,
        augmentation,
    ],
    ignore_index=True,
)






















    
    validation = load_jsonl("validation.jsonl")
    test = load_jsonl("test.jsonl")

    print()
    print(f"Train rows      : {len(train)}")
    print(f"Validation rows : {len(validation)}")
    print(f"Test rows       : {len(test)}")

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
    print("Training classifier...")

    model.fit(
        train["text"],
        train["label"],
    )

    print("Training complete.")

    print()
    print("VALIDATION RESULTS")
    print("=" * 60)

    validation_pred = model.predict(
        validation["text"]
    )

    validation_prob = model.predict_proba(
        validation["text"]
    )[:, 1]

    print(
        classification_report(
            validation["label"],
            validation_pred,
            target_names=[
                "SAFE",
                "SCAM",
            ],
        )
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                validation["label"],
                validation_prob,
            ),
            4,
        ),
    )

    print()
    print("TEST RESULTS")
    print("=" * 60)

    test_pred = model.predict(
        test["text"]
    )

    test_prob = model.predict_proba(
        test["text"]
    )[:, 1]

    accuracy = accuracy_score(
        test["label"],
        test_pred,
    )

    precision = precision_score(
        test["label"],
        test_pred,
    )

    recall = recall_score(
        test["label"],
        test_pred,
    )

    f1 = f1_score(
        test["label"],
        test_pred,
    )

    auc = roc_auc_score(
        test["label"],
        test_prob,
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    print()
    print("CONFUSION MATRIX")
    print(confusion_matrix(
        test["label"],
        test_pred,
    ))

    print()
    print("CLASSIFICATION REPORT")
    print(
        classification_report(
            test["label"],
            test_pred,
            target_names=[
                "SAFE",
                "SCAM",
            ],
        )
    )

    model_path = (
        MODEL_DIR
        / "scam_classifier.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print()
    print(
        f"Saved model → {model_path}"
    )

    print()
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    examples = [
        "Please tell me the verification code that appeared on your phone.",
        "Never share your security code with anyone, including me.",
        "Install AnyDesk so I can connect to your computer immediately.",
        "Your appointment is confirmed for tomorrow afternoon.",
    ]

    probabilities = model.predict_proba(
        examples
    )[:, 1]

    for text, probability in zip(
        examples,
        probabilities,
    ):
        print()
        print(text)
        print(
            f"Scam score: {probability:.3f}"
        )


if __name__ == "__main__":
    main()
