import numpy as np
import string
import asyncio
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ------------------------------
# Memory storage
# ------------------------------
data = {}
conversation_history = []

# Example training data
training_data = [
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey", "greeting"),
    ("bye", "farewell"),
    ("goodbye", "farewell"),
    ("see you later", "farewell"),
    ("how are you", "status"),
    ("what’s up", "status"),
    ("tell me a joke", "joke"),
]

responses = {
    "greeting": ["Hello!", "Hi there!", "Hey! How’s it going?"],
    "farewell": ["Goodbye!", "See you soon!", "Take care!"],
    "status": ["I’m doing great, thanks for asking!", "I’m just a bot, but I’m good!"],
    "joke": ["Why don’t scientists trust atoms? Because they make up everything!"],
}

# Track used responses to avoid repeats
used_responses = {intent: [] for intent in responses}

# ------------------------------
# Save & load learned data
# ------------------------------
def saveToFile(filename="data.json"):
    with open(filename, "w") as f:
        json.dump({"data": data, "responses": responses}, f)

def loadFromFile(filename="data.json"):
    global data, responses, used_responses
    try:
        with open(filename, "r") as f:
            saved = json.load(f)
            data = saved.get("data", {})
            responses.update(saved.get("responses", {}))
            # Reset usage tracking when loading
            used_responses = {intent: [] for intent in responses}
    except FileNotFoundError:
        data = {}

# ------------------------------
# Machine learning setup
# ------------------------------
X_train = [text for text, label in training_data]
y_train = [label for text, label in training_data]

vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X_train)

clf = MultinomialNB()
clf.fit(X_vectors, y_train)

# ------------------------------
# Response generation
# ------------------------------
def generateNewResponse(intent, query):
    """Generate a new response if the AI wants to be creative"""
    templates = {
        "greeting": [
            f"Nice to meet you!",
            f"Hello, {random.choice(['friend','human','there'])}!",
            f"Hey! I was just thinking about greetings.",
        ],
        "farewell": [
            f"See you later!",
            f"Goodbye and take care.",
            f"I hope we chat again soon!",
        ],
        "status": [
            f"I’m feeling {random.choice(['awesome','curious','ready to learn'])} today!",
            f"My circuits are buzzing happily.",
            f"I’m in a good mood!",
        ],
        "joke": [
            f"Here’s one: Why did the computer go to the doctor? Because it caught a bug!",
            f"I’ve got a joke: Parallel lines have so much in common… it’s a shame they’ll never meet.",
        ],
    }

    if intent in templates:
        reply = random.choice(templates[intent])
        responses[intent].append(reply)  # Save it to memory for future use
        return reply
    else:
        # Fallback: invent something from the query
        reply = f"That’s interesting! You mentioned '{random.choice(query.split())}'. Tell me more."
        responses.setdefault(intent, []).append(reply)
        return reply

def chooseResponse(intent, query):
    if intent not in responses or not responses[intent]:
        return generateNewResponse(intent, query)

    available = [r for r in responses[intent] if r not in used_responses[intent]]

    if not available:
        used_responses[intent] = []
        available = responses[intent][:]

    if random.random() < 0.3:  
        # 30% chance: invent something new
        reply = generateNewResponse(intent, query)
    else:
        reply = random.choice(available)

    used_responses[intent].append(reply)
    return reply

def generateResponse(query):
    X_test = vectorizer.transform([query])
    predicted_label = clf.predict(X_test)[0]

    reply = chooseResponse(predicted_label, query)

    # Add memory flavor
    if conversation_history:
        last_user, last_ai = conversation_history[-1]
        if any(word in query.lower() for word in last_user.split()):
            reply += " (That reminds me of earlier!)"

    # Save conversation
    conversation_history.append((query, reply))
    return reply

# ------------------------------
# Main loop
# ------------------------------
async def main():
    loadFromFile()
    print("🤖 AI Assistant Ready! Type 'quit' to exit.\n")
    while True:
        i = input("> ")
        if i.lower() in {"quit", "exit"}:
            saveToFile()
            print("Learning saved. Goodbye!")
            break
        response = generateResponse(i)
        print("AI:", response)

asyncio.run(main())
