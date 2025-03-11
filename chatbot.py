import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.join(os.path.dirname(__file__), "intents.json")
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

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    # Sidebar Menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if chat log exists, create if not
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history available.")

    # About Menu
    elif choice == "About":
        st.write("""
        The goal of this project is to create a chatbot that can understand and respond to user input based on intents.
        The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents
        and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.
        """)
        
        st.subheader("Project Overview:")
        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm are used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface.
        """)
        
        st.subheader("Dataset:")
        st.write("""
        The dataset used in this project is a collection of labeled intents and entities:
        - **Intents:** The intent of the user input (e.g. "greeting", "budget", "about").
        - **Entities:** The extracted entities from user input (e.g. "Hi", "How do I create a budget?").
        - **Text:** The user input text.
        """)
        
        st.subheader("Streamlit Chatbot Interface:")
        st.write("""
        The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window
        to display the chatbot's responses. The interface uses the trained model to generate responses to user input.
        """)
        
        st.subheader("Conclusion:")
        st.write("""
        In this project, a chatbot is built that can understand and respond to user input based on intents.
        The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit.
        This project can be extended by adding more data, using more sophisticated NLP techniques, and deep learning algorithms.
        """)

if __name__ == '__main__':
    main()