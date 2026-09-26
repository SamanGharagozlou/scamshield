from fastapi.testclient import TestClient

from scamshield.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "ok"

    assert (
        result["service"]
        == "scamshield-detection"
    )


def test_detect_endpoint_accumulates_conversation_risk():
    conversation_id = "api-bank-scam"

    client.delete(
        f"/conversations/{conversation_id}"
    )

    response_1 = client.post(
        "/detect",
        json={
            "conversation_id": conversation_id,
            "timestamp": 1,
            "speaker": "caller",
            "text": (
                "Hello, I'm calling from your bank."
            ),
        },
    )

    assert response_1.status_code == 200

    response_2 = client.post(
        "/detect",
        json={
            "conversation_id": conversation_id,
            "timestamp": 2,
            "speaker": "caller",
            "text": (
                "We need to fix this immediately."
            ),
        },
    )

    assert response_2.status_code == 200

    response_3 = client.post(
        "/detect",
        json={
            "conversation_id": conversation_id,
            "timestamp": 3,
            "speaker": "caller",
            "text": (
                "Please read the code to me."
            ),
        },
    )

    assert response_3.status_code == 200

    result = response_3.json()

    assert (
        result["risk_level"]
        == "CRITICAL"
    )

    assert result["risk_score"] >= 80

    assert (
        result["recommended_action"]
        == "DO_NOT_SHARE_CODE"
    )

    assert (
        result["scam_category"]
        == "BANK_IMPERSONATION"
    )

    assert (
        result["ml_scam_score"]
        is not None
    )

    assert (
        result["detection_mode"]
        == "hybrid"
    )


def test_reset_endpoint_clears_conversation():
    conversation_id = "api-reset-test"

    client.delete(
        f"/conversations/{conversation_id}"
    )

    client.post(
        "/detect",
        json={
            "conversation_id": conversation_id,
            "timestamp": 1,
            "speaker": "caller",
            "text": (
                "Hello, I'm calling from your bank."
            ),
        },
    )

    reset_response = client.delete(
        f"/conversations/{conversation_id}"
    )

    assert (
        reset_response.status_code
        == 200
    )

    reset_result = (
        reset_response.json()
    )

    assert (
        reset_result["status"]
        == "reset"
    )

    response = client.post(
        "/detect",
        json={
            "conversation_id": conversation_id,
            "timestamp": 2,
            "speaker": "caller",
            "text": (
                "Hello, how are you today?"
            ),
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert (
        result["risk_level"]
        == "SAFE"
    )

    assert (
        result["risk_score"]
        < 30
    )

    assert result["signals"] == []


def test_detect_returns_hybrid_fields():
    conversation_id = "api-hybrid-fields"

    client.delete(
        f"/conversations/{conversation_id}"
    )

    response = client.post(
        "/detect",
        json={
            "conversation_id": conversation_id,
            "timestamp": 1,
            "speaker": "caller",
            "text": (
                "Your appointment is confirmed "
                "for tomorrow afternoon."
            ),
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert "ml_scam_score" in result

    assert (
        result["ml_scam_score"]
        is not None
    )

    assert (
        0.0
        <= result["ml_scam_score"]
        <= 1.0
    )

    assert (
        result["detection_mode"]
        == "hybrid"
    )
