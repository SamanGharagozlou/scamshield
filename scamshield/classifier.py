from pathlib import Path

import joblib


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "scam_classifier.joblib"
)


class SemanticScamClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Scam classifier model not found: {model_path}"
            )

        self.model = joblib.load(model_path)

    def predict_score(self, text: str) -> float:
        """
        Return a scam score between 0 and 1.

        0.0 = model considers the text more likely legitimate
        1.0 = model considers the text more likely scam
        """
        if not text.strip():
            return 0.0

        probabilities = self.model.predict_proba([text])[0]

        classes = list(self.model.classes_)

        scam_index = classes.index(1)

        return float(probabilities[scam_index])

    def predict_label(
        self,
        text: str,
        threshold: float = 0.5,
    ) -> str:
        score = self.predict_score(text)

        return "SCAM" if score >= threshold else "SAFE"
