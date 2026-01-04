import os
import json
import csv
import random
import datetime
import streamlit as st
import pyttsx3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ================== CONFIG ==================
INTENTS_FILE = "intents.json"
LEARNED_FILE = "learned_knowledge.json"
CONFIDENCE_THRESHOLD = 0.55

# ================== LOAD INTENTS ==================
with open(INTENTS_FILE, "r", encoding="utf-8") as f:
    intents = json.load(f)

# ================== TRAIN MODEL (ONCE) ==================
@st.cache_resource
def train_model():
    patterns = []
    tags = []

    for intent in intents:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            tags.append(intent["tag"])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(patterns)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, tags)

    return vectorizer, model

vectorizer, model = train_model()

# ================== TEXT TO SPEECH ==================
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# ================== LEARNED KNOWLEDGE ==================
def load_learned_data():
    if not os.path.exists(LEARNED_FILE):
        return {}
    with open(LEARNED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_learned_data(data):
    with open(LEARNED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ================== CHATBOT RESPONSE ==================
def chatbot_response(user_text):
    clean_text = user_text.strip().lower()

    # 1️⃣ Learned knowledge first
    learned_data = load_learned_data()
    if clean_text in learned_data:
        return learned_data[clean_text]

    # 2️⃣ Predict intent
    vec = vectorizer.transform([user_text])
    probs = model.predict_proba(vec)[0]
    confidence = max(probs)
    tag = model.predict(vec)[0]

    # 3️⃣ SYSTEM INTENTS → ALWAYS ANSWER
    system_intents = ["greeting", "goodbye", "about_bot"]

    if tag in system_intents:
        for intent in intents:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

    # 4️⃣ DOMAIN INTENTS → APPLY CONFIDENCE
    if confidence < CONFIDENCE_THRESHOLD:
        st.session_state["awaiting_learning"] = clean_text
        return (
            "I’m not sure about this yet. "
            "If you know the correct answer, you can tell me and I’ll remember it."
        )

    # 5️⃣ Normal intent response
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I currently focus on smart farming topics."

# ================== LOG CONVERSATION ==================
def log_chat(user, bot):
    if not os.path.exists("chat_log.csv"):
        with open("chat_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["User", "Bot", "Time"])

    with open("chat_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user, bot, datetime.datetime.now()])

# ================== STREAMLIT UI ==================
def main():
    st.set_page_config(page_title="Smart Farming Chatbot", layout="centered")
    st.title("🤖 Smart Farming Chatbot")

    menu = ["Chat", "History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "awaiting_learning" not in st.session_state:
        st.session_state.awaiting_learning = None

    # -------- CHAT PAGE --------
    if choice == "Chat":

        use_voice = st.checkbox("🔊 Voice Reply")

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("You:")
            submitted = st.form_submit_button("Send")

        if submitted and user_input:
            bot_reply = chatbot_response(user_input)

            st.session_state.messages.append(("You", user_input))
            st.session_state.messages.append(("Bot", bot_reply))

            log_chat(user_input, bot_reply)

            if use_voice:
                speak(bot_reply)

        # -------- LEARNING MODE --------
        if st.session_state.awaiting_learning:
            st.info("🧠 Learning mode: Please provide the correct answer.")

            learned_answer = st.text_input("Your answer:")

            if st.button("Teach the bot"):
                data = load_learned_data()
                data[st.session_state.awaiting_learning] = learned_answer
                save_learned_data(data)

                st.session_state.messages.append(
                    ("Bot", "Thank you! I have learned this and will remember it. ✅")
                )

                st.session_state.awaiting_learning = None

        # -------- DISPLAY CHAT --------
        for sender, msg in st.session_state.messages:
            st.write(f"**{sender}:** {msg}")

    # -------- HISTORY PAGE --------
    elif choice == "History":
        st.header("Conversation History")
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    st.write(f"👤 {row[0]}")
                    st.write(f"🤖 {row[1]}")
                    st.caption(row[2])
                    st.divider()
        else:
            st.info("No history available.")

    # -------- ABOUT PAGE -------- 
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
