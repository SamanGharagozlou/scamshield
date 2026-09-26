from pathlib import Path

from datasets import load_dataset


OUTPUT_DIR = Path("data/external")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading external phone-scam dataset...")

    dataset = load_dataset(
        "menaattia/phone-scam-dataset"
    )

    print(dataset)

    if "test" not in dataset:
        raise RuntimeError(
            "Expected a test split but none was found."
        )

    test = dataset["test"]

    print()
    print("External test rows:", len(test))
    print("Columns:", test.column_names)

    output = OUTPUT_DIR / "menaattia_test.jsonl"

    test.to_json(str(output))

    print()
    print(f"Saved → {output}")


if __name__ == "__main__":
    main()
