from fastapi.testclient import TestClient

from scamshield.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "scamshield-detection",
    }


def test_detect_endpoint_accumulates_conversation_risk():
    conversation_id = "api-bank-scam"

    messages = [
        {
            "timestamp": 1,
            "text": "Hello, I'm calling from your bank.",
        },
        {
            "timestamp": 2,
            "text": "We need to fix this immediately.",
        },
        {
            "timestamp": 3,
            "text": "Please read the code to me.",
        },
    ]

    response = None

    for message in messages:
        response = client.post(
            "/detect",
            json={
                "conversation_id": conversation_id,
                "timestamp": message["timestamp"],
                "speaker": "caller",
                "text": message["text"],
            },
        )

        assert response.status_code == 200

    result = response.json()

    assert result["risk_level"] == "CRITICAL"
    assert result["scam_category"] == "BANK_IMPERSONATION"
    assert result["recommended_action"] == "DO_NOT_SHARE_CODE"
    assert "BANK_CLAIM" in result["signals"]
    assert "OTP_REQUEST" in result["signals"]


def test_reset_endpoint_clears_conversation():
    conversation_id = "api-reset-test"

    client.post(
        "/detect",
        json={
            "conversation_id": conversation_id,
            "timestamp": 1,
            "speaker": "caller",
            "text": "Hello, I'm calling from your bank.",
        },
    )

    reset_response = client.delete(
        f"/conversations/{conversation_id}"
    )

    assert reset_response.status_code == 200

    response = client.post(
        "/detect",
        json={
            "conversation_id": conversation_id,
            "timestamp": 2,
            "speaker": "caller",
            "text": "Hello, how are you today?",
        },
    )

    result = response.json()

    assert result["risk_score"] == 0
    assert result["risk_level"] == "SAFE"
    assert result["signals"] == []
