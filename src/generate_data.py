import pandas as pd
from itertools import product

ticket_types = {
    "Billing": [
        "I was charged twice for my monthly subscription",
        "My invoice shows an incorrect amount",
        "I need a refund for my last payment",
        "My payment failed but money was deducted",
        "Please explain the extra charge on my account",
    ],
    "Technical Support": [
        "The application crashes when I open it",
        "I cannot upload my file to the application",
        "The website keeps showing an error",
        "The application is running very slowly",
        "I cannot connect to the service",
    ],
    "Account": [
        "I cannot log in to my account",
        "Please help me reset my password",
        "My account has been locked",
        "I need to update my email address",
        "I cannot verify my account",
    ],
    "Delivery": [
        "My order has not arrived yet",
        "The tracking page shows no updates",
        "I received the wrong product",
        "My package arrived damaged",
        "Please tell me the status of my order",
    ],
}

priority_phrases = {
    "Low": [
        "This is not urgent.",
        "Please help when convenient.",
        "There is no immediate impact.",
        "I can wait for a response.",
    ],
    "Medium": [
        "Please resolve this soon.",
        "This is affecting my normal work.",
        "I need help within a reasonable time.",
        "The issue is causing some inconvenience.",
    ],
    "High": [
        "This is urgent and needs immediate attention.",
        "I cannot continue my work.",
        "This is causing a serious disruption.",
        "Please resolve this as soon as possible.",
    ],
}

rows = []

for category, priority, number in product(
    ticket_types.keys(),
    priority_phrases.keys(),
    range(50),
):
    issue = ticket_types[category][number % 5]
    urgency = priority_phrases[priority][number % 4]

    ticket_text = (
        f"{issue}. {urgency} "
        f"Reference number {number + 1}."
    )

    rows.append({
        "text": ticket_text,
        "category": category,
        "priority": priority,
    })

df = pd.DataFrame(rows)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/tickets.csv", index=False)

print("Dataset created successfully")
print("Dataset shape:", df.shape)
print("\nCategory counts:")
print(df["category"].value_counts())
print("\nPriority counts:")
print(df["priority"].value_counts())
