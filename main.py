from fastapi import FastAPI, Request
from pydantic import BaseModel
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize app
app = FastAPI()

# NLP Setup
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('model.h5')

# Input schema
class Message(BaseModel):
    message: str

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence, threshold=0.25):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[i], "probability": float(p)} for i, p in results]

def get_response(ints, intents_json):
    if not ints:
        return "Sorry, I didn't quite understand that. Try asking something else."
    tag = ints[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

@app.post("/chat")
def chat(msg: Message):
    intents_list = predict_class(msg.message)
    response = get_response(intents_list, intents)
    return {"response": response}
