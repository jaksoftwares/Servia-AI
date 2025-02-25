import json
import random
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_training_data():
    """Loads training data from training_data.json"""
    with open(os.path.join(BASE_DIR, "training_data.json"), "r") as file:
        return json.load(file)

def train_intent_classifier():
    """Trains a chatbot intent classifier"""
    data = load_training_data()
    
    X_texts, y_labels = [], []

    for intent in data["intents"]:  # Loop through the intents list
        for pattern in intent["patterns"]:  # Use "patterns" instead of "examples"
            X_texts.append(pattern)
            y_labels.append(intent["tag"])  # Use the tag as the label

    # Convert text into numerical data
    vectorizer = CountVectorizer()
    X_vectors = vectorizer.fit_transform(X_texts)
    
    # Train a simple classifier
    model = LogisticRegression()
    model.fit(X_vectors, y_labels)

    # Save the trained model
    with open(os.path.join(BASE_DIR, "intent_classifier.pkl"), "wb") as f:
        pickle.dump((vectorizer, model), f)

    print("✅ Chatbot AI model trained successfully!")

if __name__ == "__main__":
    train_intent_classifier()
