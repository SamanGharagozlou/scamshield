from scamshield.classifier import SemanticScamClassifier


def test_classifier_returns_valid_probability():
    classifier = SemanticScamClassifier()

    score = classifier.predict_score(
        "Please tell me the verification code."
    )

    assert 0.0 <= score <= 1.0


def test_empty_text_returns_zero():
    classifier = SemanticScamClassifier()

    assert classifier.predict_score("") == 0.0
