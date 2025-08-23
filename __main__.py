import numpy as np
import string
import asyncio
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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

used_responses = {intent: [] for intent in responses}

# Word definitions for keyword-based fallback
word_definitions = {
    "python": "Python is a programming language that’s great for automation and AI.",
    "robot": "A robot is a machine capable of carrying out tasks automatically.",
    "hello": "Hello is a greeting used when meeting someone.",
    "hi": "Hi is a casual way to greet someone.",
    "bye": "Bye is a word used to say farewell.",
    "joke": "A joke is a funny story or statement meant to make people laugh.",
    "ai": "AI stands for artificial intelligence, machines simulating human intelligence.",
}

# ------------------------------
# File persistence
# ------------------------------
def saveToFile(filename="data.json"):
    with open(filename, "w") as f:
        json.dump({
            "data": data,
            "responses": responses,
            "history": conversation_history,
            "definitions": word_definitions
        }, f)

def loadFromFile(filename="data.json"):
    global data, responses, used_responses, conversation_history, word_definitions
    try:
        with open(filename, "r") as f:
            saved = json.load(f)
            data = saved.get("data", {})
            responses.update(saved.get("responses", {}))
            conversation_history.extend(saved.get("history", []))
            word_definitions.update(saved.get("definitions", {}))
            used_responses = {intent: [] for intent in responses}
    except FileNotFoundError:
        data, conversation_history = {}, {}

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
        responses[intent].append(reply)
        return reply
    else:
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
        reply = generateNewResponse(intent, query)
    else:
        reply = random.choice(available)

    used_responses[intent].append(reply)
    return reply

def contextAwareFlavor(query):
    """Adds context awareness by checking last 3 messages"""
    if not conversation_history:
        return ""
    recent_text = " ".join([msg for msg, _ in conversation_history[-3:]])
    recent_words = set(recent_text.lower().split())
    if any(word in query.lower().split() for word in recent_words):
        return " (That connects to what you said earlier!)"
    return ""

def learnNewDefinition(word):
    """Ask the user for a definition of an unknown word and return a contextual response"""
    print(f"🤖 I don't know the word '{word}'. Can you provide a short definition?")
    definition = input("You: ")
    word_definitions[word] = definition
    saveToFile()
    # Generate a natural response including the definition
    reply = f"Got it! So '{word}' means: {definition}. Thanks for teaching me!"
    return reply

def keywordFallback(query):
    """Generates a fallback using keywords and definitions, or learns new words in context"""
    words = query.lower().translate(str.maketrans('', '', string.punctuation)).split()
    known_defs = [f"'{word}' means: {word_definitions[word]}" for word in words if word in word_definitions]
    
    if known_defs:
        return f"Interesting! {random.choice(known_defs)}"
    
    unknown_words = [word for word in words if word not in word_definitions]
    if unknown_words:
        return learnNewDefinition(unknown_words[0])
    
    fallbacks = [
        "Hmm, I’m not sure what you mean. Can you rephrase?",
        "I didn’t quite get that. Could you say it differently?",
        "That’s interesting… can you clarify?",
        "I’m not certain I understand. Tell me more!"
    ]
    return random.choice(fallbacks)

def generateResponse(query):
    X_test = vectorizer.transform([query])
    probs = clf.predict_proba(X_test)[0]
    confidence = max(probs)
    predicted_label = clf.classes_[np.argmax(probs)]

    if confidence < 0.4:
        reply = keywordFallback(query)
    else:
        reply = chooseResponse(predicted_label, query)
        reply += contextAwareFlavor(query)

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
