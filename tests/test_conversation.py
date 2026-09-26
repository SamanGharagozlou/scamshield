from scamshield.conversation import ConversationState
from scamshield.signals import ScamSignal


def test_conversation_stores_transcript():
    state = ConversationState(conversation_id="test-1")

    state.add_text("Hello")
    state.add_text("I'm calling from your bank.")

    assert state.transcript == [
        "Hello",
        "I'm calling from your bank.",
    ]


def test_conversation_accumulates_signals():
    state = ConversationState(conversation_id="test-2")

    state.add_signals([ScamSignal.BANK_CLAIM])
    state.add_signals([ScamSignal.URGENCY])

    assert ScamSignal.BANK_CLAIM in state.signals
    assert ScamSignal.URGENCY in state.signals


def test_duplicate_signals_are_not_repeated():
    state = ConversationState(conversation_id="test-3")

    state.add_signals([ScamSignal.BANK_CLAIM])
    state.add_signals([ScamSignal.BANK_CLAIM])

    assert len(state.signals) == 1


def test_recent_context():
    state = ConversationState(conversation_id="test-4")

    for i in range(10):
        state.add_text(f"message-{i}")

    context = state.recent_context(limit=3)

    assert context == [
        "message-7",
        "message-8",
        "message-9",
    ]


def test_reset():
    state = ConversationState(conversation_id="test-5")

    state.add_text("Suspicious message")
    state.add_signals([ScamSignal.URGENCY])
    state.risk_score = 50
    state.risk_level = "HIGH"

    state.reset()

    assert state.transcript == []
    assert state.signals == set()
    assert state.risk_score == 0.0
    assert state.risk_level == "SAFE"
