import json
from pathlib import Path


OUTPUT = Path("data/augmentation/scamshield_train.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


examples = [
    # --------------------------------------------------
    # GOVERNMENT — SAFE
    # --------------------------------------------------

    {
        "text": "This is an informational call from the Social Security office. We will never ask you to provide your Social Security number, password, or payment information over the phone.",
        "label": 0,
        "category": "SAFE_GOVERNMENT",
    },
    {
        "text": "For security, please do not give your Social Security number to anyone who calls you. Visit the official government website yourself if you have questions.",
        "label": 0,
        "category": "SAFE_GOVERNMENT",
    },
    {
        "text": "The agency has mailed you a notice. You do not need to make a payment during this call and we will not ask for banking details.",
        "label": 0,
        "category": "SAFE_GOVERNMENT",
    },

    # GOVERNMENT — SCAM
    {
        "text": "This is the Social Security Administration. Your number has been suspended and you must confirm your Social Security number immediately.",
        "label": 1,
        "category": "GOVERNMENT_SCAM",
    },
    {
        "text": "There is a warrant connected with your identity. Give me your personal information now or officers may be sent to your home.",
        "label": 1,
        "category": "GOVERNMENT_SCAM",
    },
    {
        "text": "You owe unpaid taxes and must transfer the money today to prevent legal action.",
        "label": 1,
        "category": "GOVERNMENT_SCAM",
    },

    # --------------------------------------------------
    # TECH SUPPORT — SAFE
    # --------------------------------------------------

    {
        "text": "I can guide you through the settings, but I will not ask to remotely control your computer.",
        "label": 0,
        "category": "SAFE_TECH",
    },
    {
        "text": "Please do not install remote access software such as AnyDesk or TeamViewer because of this phone call.",
        "label": 0,
        "category": "SAFE_TECH",
    },
    {
        "text": "You opened a support ticket yesterday. I can explain the troubleshooting steps, but you should keep control of your computer.",
        "label": 0,
        "category": "SAFE_TECH",
    },

    # TECH SUPPORT — SCAM
    {
        "text": "Your computer is infected. Download AnyDesk now and give me the code so I can connect.",
        "label": 1,
        "category": "TECH_SUPPORT_SCAM",
    },
    {
        "text": "Open the remote assistance application and allow me to control your computer so I can remove the virus.",
        "label": 1,
        "category": "TECH_SUPPORT_SCAM",
    },
    {
        "text": "There is a security problem on your laptop. Install this support program immediately and tell me the connection number.",
        "label": 1,
        "category": "TECH_SUPPORT_SCAM",
    },

    # --------------------------------------------------
    # BANK — SAFE
    # --------------------------------------------------

    {
        "text": "You may receive a security code when signing in. Never share that code with me or anyone else.",
        "label": 0,
        "category": "SAFE_BANK",
    },
    {
        "text": "Your bank will never ask for your password, PIN, or one-time verification code during a phone call.",
        "label": 0,
        "category": "SAFE_BANK",
    },
    {
        "text": "Please open the official banking application yourself. I do not need your PIN or security code.",
        "label": 0,
        "category": "SAFE_BANK",
    },

    # BANK — SCAM
    {
        "text": "I'm calling from account security. Tell me the six digits that were just sent to your phone.",
        "label": 1,
        "category": "BANK_OTP",
    },
    {
        "text": "We detected fraud on your account. What numbers appeared in the verification message you just received?",
        "label": 1,
        "category": "BANK_OTP",
    },
    {
        "text": "To cancel the suspicious payment I need you to read the security code displayed on your phone.",
        "label": 1,
        "category": "BANK_OTP",
    },

    # --------------------------------------------------
    # DELIVERY — SAFE
    # --------------------------------------------------

    {
        "text": "Your package will arrive tomorrow. You can track it in the official delivery application and no payment is required.",
        "label": 0,
        "category": "SAFE_DELIVERY",
    },
    {
        "text": "There has been a delivery delay. Check the tracking number on our official website when convenient.",
        "label": 0,
        "category": "SAFE_DELIVERY",
    },
    {
        "text": "Your parcel is ready for collection. Please bring identification to the collection point.",
        "label": 0,
        "category": "SAFE_DELIVERY",
    },

    # DELIVERY — SCAM
    {
        "text": "Your parcel is being held. Give me your card number now to pay the customs fee.",
        "label": 1,
        "category": "DELIVERY_SCAM",
    },
    {
        "text": "We cannot release your package until you pay a small charge during this phone call.",
        "label": 1,
        "category": "DELIVERY_SCAM",
    },
    {
        "text": "Your delivery requires an immediate payment. Tell me your credit card details so I can release it.",
        "label": 1,
        "category": "DELIVERY_SCAM",
    },

    # --------------------------------------------------
    # REFUND — SAFE
    # --------------------------------------------------

    {
        "text": "Your refund will automatically be returned to your original payment method. You do not need to send us any money.",
        "label": 0,
        "category": "SAFE_REFUND",
    },
    {
        "text": "The refund has already been processed. We will never ask you to return money by bank transfer.",
        "label": 0,
        "category": "SAFE_REFUND",
    },
    {
        "text": "You can check the refund status in your account. Do not provide payment details during this call.",
        "label": 0,
        "category": "SAFE_REFUND",
    },

    # REFUND — SCAM
    {
        "text": "We accidentally refunded too much money. You must send the extra amount back immediately.",
        "label": 1,
        "category": "REFUND_SCAM",
    },
    {
        "text": "There was an error with your refund. Transfer the excess money back today or your account will be blocked.",
        "label": 1,
        "category": "REFUND_SCAM",
    },
    {
        "text": "Your refund was processed incorrectly and I need you to return part of it by bank transfer.",
        "label": 1,
        "category": "REFUND_SCAM",
    },

    # --------------------------------------------------
    # FAMILY — SAFE
    # --------------------------------------------------

    {
        "text": "Hi Grandma, it's me. Nothing is wrong. I just wanted to call and see how you're doing.",
        "label": 0,
        "category": "SAFE_FAMILY",
    },
    {
        "text": "I'll call you again tonight. There's no emergency and you don't need to send anything.",
        "label": 0,
        "category": "SAFE_FAMILY",
    },

    # FAMILY — SCAM
    {
        "text": "Grandma, I had an accident and need money immediately. Please don't tell my parents.",
        "label": 1,
        "category": "FAMILY_EMERGENCY",
    },
    {
        "text": "I'm calling from a friend's phone. I need you to send me money today and please keep this between us.",
        "label": 1,
        "category": "FAMILY_EMERGENCY",
    },

    # --------------------------------------------------
    # INVESTMENT
    # --------------------------------------------------

    {
        "text": "Investments involve risk and returns are never guaranteed. Take time to review the documents before deciding.",
        "label": 0,
        "category": "SAFE_INVESTMENT",
    },
    {
        "text": "This investment guarantees twenty percent profit this month. Transfer your money today before the opportunity closes.",
        "label": 1,
        "category": "INVESTMENT_SCAM",
    },

    # --------------------------------------------------
    # PASSWORD / PIN
    # --------------------------------------------------

    {
        "text": "Never tell me your password or PIN. I do not need either of them to help you.",
        "label": 0,
        "category": "SAFE_SECURITY",
    },
    {
        "text": "Tell me your account password so I can verify that you are the owner.",
        "label": 1,
        "category": "PASSWORD_THEFT",
    },
]


with OUTPUT.open("w", encoding="utf-8") as f:
    for index, example in enumerate(examples):
        record = {
            "id": f"scamshield_aug_{index:04d}",
            **example,
            "source": "scamshield_curated",
        }

        f.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )


safe = sum(x["label"] == 0 for x in examples)
scam = sum(x["label"] == 1 for x in examples)

print(f"Created {len(examples)} augmentation examples")
print(f"SAFE: {safe}")
print(f"SCAM: {scam}")
print(f"Saved → {OUTPUT}")
