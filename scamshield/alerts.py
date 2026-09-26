import json
import os
import urllib.error
import urllib.request


class CriticalAlertManager:
    """
    Sends CRITICAL ScamShield events to n8n.

    - Only CRITICAL events are sent.
    - Only one alert is sent per conversation.
    - If the webhook fails, the conversation is NOT marked
      as alerted, so a later CRITICAL event can retry.
    """

    def __init__(self):
        self.alerted_conversations: set[str] = set()

    @property
    def webhook_url(self) -> str | None:
        return os.getenv("N8N_WEBHOOK_URL")

    def send_if_needed(
        self,
        risk_event,
        latest_text: str,
    ) -> bool:

        # Only send trusted-contact alerts for CRITICAL risk.
        if risk_event.risk_level != "CRITICAL":
            return False

        conversation_id = risk_event.conversation_id

        # Avoid repeatedly alerting the family member.
        if conversation_id in self.alerted_conversations:
            return False

        webhook_url = self.webhook_url

        if not webhook_url:
            print(
                "[ScamShield] CRITICAL detected, "
                "but N8N_WEBHOOK_URL is not configured."
            )
            return False

        payload = {
            "event": "scamshield_critical_alert",
            "conversation_id": conversation_id,
            "timestamp": risk_event.timestamp,
            "risk_level": risk_event.risk_level,
            "risk_score": risk_event.risk_score,
            "scam_category": risk_event.scam_category,
            "signals": risk_event.signals,
            "recommended_action": risk_event.recommended_action,

            # Only send the latest detected phrase rather than
            # the entire private conversation.
            "latest_text": latest_text,
        }

        encoded_payload = json.dumps(
            payload
        ).encode("utf-8")

        request = urllib.request.Request(
            webhook_url,
            data=encoded_payload,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=3,
            ) as response:

                response.read()

            self.alerted_conversations.add(
                conversation_id
            )

            print()
            print(
                "📱 Trusted-contact alert sent through n8n."
            )

            return True

        except (
            urllib.error.URLError,
            TimeoutError,
            Exception,
        ) as exc:

            print()
            print(
                f"⚠️ n8n alert failed: {exc}"
            )

            return False

    def reset(
        self,
        conversation_id: str,
    ) -> None:

        self.alerted_conversations.discard(
            conversation_id
        )


critical_alerts = CriticalAlertManager()
