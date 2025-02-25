from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import random
import json
import os

# Initialize FastAPI
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and vectorizer
with open(os.path.join(BASE_DIR, "intent_classifier.pkl"), "rb") as f:
    vectorizer, model = pickle.load(f)

# Load the training data
def load_training_data():
    with open(os.path.join(BASE_DIR, "training_data.json"), "r") as file:
        data = json.load(file)
    
    # Convert intents into a dictionary for response lookup
    return {intent["tag"]: intent["responses"] for intent in data["intents"]}

responses = load_training_data()

# Define request body model
class UserInput(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Servia AI Chatbot API is running!"}

@app.post("/chat")
def chat(user_input: UserInput):
    """Receives user input, predicts intent, and returns a chatbot response."""
    X_vector = vectorizer.transform([user_input.message])
    predicted_intent = model.predict(X_vector)[0]
    
    bot_response = random.choice(responses.get(predicted_intent, ["Sorry, I don't understand."]))
    return {"response": bot_response}

