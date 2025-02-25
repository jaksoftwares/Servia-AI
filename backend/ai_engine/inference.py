import json
import pickle
import random
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model and vectorizer
with open(os.path.join(BASE_DIR, "intent_classifier.pkl"), "rb") as f:
    vectorizer, model = pickle.load(f)

def load_training_data():
    """Loads training data and converts it into a dictionary of responses"""
    with open(os.path.join(BASE_DIR, "training_data.json"), "r") as file:
        data = json.load(file)
    
    # Convert the list of intents into a dictionary
    responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}
    return responses

responses = load_training_data()

def get_chatbot_response(user_input):
    """Predicts intent and returns a random response from the corresponding intent"""
    X_vector = vectorizer.transform([user_input])
    predicted_intent = model.predict(X_vector)[0]

    # Get a response from the predicted intent
    return random.choice(responses.get(predicted_intent, ["Sorry, I don't understand."]))

# Example test
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        print(f"Chatbot: {get_chatbot_response(user_input)}")
