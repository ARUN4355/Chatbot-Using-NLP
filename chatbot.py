import os
import json
import datetime
import csv
import random
import nltk
import ssl
import streamlit as st
import speech_recognition as sr
import pyttsx3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL certificate issue for nltk
ssl._create_default_https_context = ssl._create_unverified_context

# Download necessary nltk resources
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from JSON file
file_path = os.path.abspath("intents.json")  # Ensure this file is in the same directory
with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Function to speak chatbot responses (fixed)
def speak(text):
    engine = pyttsx3.init()  # Reinitialize engine inside the function
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech input
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Speech recognition service is unavailable."

# Function to generate chatbot response
def chatbot(input_text):
    input_text_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_text_vector)[0]
    
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    # If chatbot has no answer, ask user
    return "I don't know the answer. Can you help me by providing one?"

# Function to log conversation
def log_conversation(user_input, response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([user_input, response, timestamp])

# Streamlit UI
def main():
    st.title("Intent-Based Chatbot using NLP 🤖")

    # Sidebar Menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot! Type or use voice input to chat.")

        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

        # Voice Toggle Button
        use_voice = st.checkbox("🔊 Enable Voice Reply")

        # Chat interface (continuous chat like real messaging)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        user_input = st.text_input("You:", key="user_input")
        voice_input = st.button("🎙 Use Voice")

        if voice_input:
            user_input = recognize_speech()

        if user_input:
            response = chatbot(user_input)
            st.session_state.messages.append(("You", user_input))
            st.session_state.messages.append(("Chatbot", response))

            # Display chat history
            for sender, message in st.session_state.messages:
                st.text(f"{sender}: {message}")

            log_conversation(user_input, response)

            if use_voice:
                speak(response)  # Speak response only if enabled

            if response.lower() in ["goodbye", "bye"]:
                st.write("Thank you for chatting! Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        except FileNotFoundError:
            st.write("No conversation history found.")

    elif choice == "About":
        st.write(
            """
            ### Goal
            The goal of this project is to create a chatbot that can understand and respond to user input based on intents.
            The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents
            and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.
            
            ### Project Overview
            The project is divided into two parts:
            1. NLP techniques and Logistic Regression algorithm are used to train the chatbot on labeled intents and entities.
            2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface.
            
            ### Dataset
            The dataset used in this project is a collection of labeled intents and entities:
            - **Intents:** The intent of the user input (e.g. "greeting", "budget", "about").
            - **Entities:** The extracted entities from user input (e.g. "Hi", "How do I create a budget?").
            - **Text:** The user input text.
            
            ### Features
            - **Text & Voice Input** (Speech recognition)
            - **Text-to-Speech Responses** (Toggle 🔊)
            - **Conversation Logging**
            - **Continuous Chat View**
            - **User Feedback for Unknown Questions**
            
            ### Conclusion
            In this project, a chatbot is built that can understand and respond to user input based on intents.
            The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit.
            This project can be extended by adding more data, using more sophisticated NLP techniques, and deep learning algorithms.
            """
        )

if __name__ == "__main__":
    main()

# To run the chatbot file enter "streamlit run chatbot.py" in the terminal..