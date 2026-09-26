from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SCAM_CATEGORY_MAP = {
    "ssn": "GOVERNMENT_IMPERSONATION",
    "refund": "REFUND_SCAM",
    "support": "TECH_SUPPORT_SCAM",
    "reward": "REWARD_SCAM",
}


def normalize_dataframe(
    df: pd.DataFrame,
    source_split: str,
) -> pd.DataFrame:
    required_columns = {
        "dialogue",
        "type",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy().reset_index(drop=True)

    df["id"] = [
        f"bothbosu_{source_split}_{i:05d}"
        for i in range(len(df))
    ]

    df["text"] = (
        df["dialogue"]
        .astype(str)
        .str.strip()
    )

    df["label"] = df["label"].astype(int)

    df["label_name"] = df["label"].map({
        0: "SAFE",
        1: "SCAM",
    })

    df["source_type"] = df["type"]

    df["scam_category"] = df.apply(
        lambda row: (
            SCAM_CATEGORY_MAP.get(row["type"])
            if row["label"] == 1
            else None
        ),
        axis=1,
    )

    df["source"] = "BothBosu/scam-dialogue"

    return df[
        [
            "id",
            "text",
            "label",
            "label_name",
            "source_type",
            "scam_category",
            "source",
        ]
    ]


def save_jsonl(
    df: pd.DataFrame,
    filename: str,
) -> None:
    path = OUTPUT_DIR / filename

    df.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print(f"Saved {len(df):4d} rows → {path}")


def main():
    train_path = RAW_DIR / "scam-dialogue_train.csv"
    test_path = RAW_DIR / "scam-dialogue_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not test_path.exists():
        raise FileNotFoundError(test_path)

    raw_train = pd.read_csv(train_path)
    raw_test = pd.read_csv(test_path)

    full_train = normalize_dataframe(
        raw_train,
        "train",
    )

    test = normalize_dataframe(
        raw_test,
        "test",
    )

    # Create validation data from the training set.
    # We sample from every conversation type so
    # validation remains balanced across categories.
    validation = (
        full_train
        .groupby(
            "source_type",
            group_keys=False,
        )
        .sample(
            frac=0.15,
            random_state=42,
        )
    )

    train = full_train.drop(
        validation.index
    )

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    save_jsonl(
        train,
        "train.jsonl",
    )

    save_jsonl(
        validation,
        "validation.jsonl",
    )

    save_jsonl(
        test,
        "test.jsonl",
    )

    print()
    print("TRAIN LABELS")
    print(train["label_name"].value_counts())

    print()
    print("VALIDATION LABELS")
    print(validation["label_name"].value_counts())

    print()
    print("TEST LABELS")
    print(test["label_name"].value_counts())

    print()
    print("TRAIN TYPES")
    print(train["source_type"].value_counts())


if __name__ == "__main__":
    main()
