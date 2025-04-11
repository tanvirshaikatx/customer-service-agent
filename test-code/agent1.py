# agent.py
import speech_recognition as sr
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pyttsx3

load_dotenv()
engine = pyttsx3.init()

# RAG Setup
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-pro")
chain = load_qa_chain(llm, chain_type="stuff")

# TTS helper
def speak(text):
    engine.say(text)
    engine.runAndWait()

# STT helper
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Listening...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

# Conversation Flow
def conversation():
    order = []
    user_address = ""

    speak("Hello, how can I help you today?")
    user_input = listen().lower()
    
    if "order" in user_input:
        speak("What would you like to order?")
        item_input = listen()
        if item_input:
            docs = vectorstore.similarity_search(item_input, k=3)
            response = chain.run(input_documents=docs, question=f"What is the price and detail of: {item_input}")
            speak(f"You ordered: {item_input}. {response}")
            
            # Assuming response includes price
            speak("Should I confirm your order?")
            confirm = listen().lower()
            if "yes" in confirm:
                speak("Please tell me your delivery address.")
                user_address = listen()
                speak("Thank you. Your order is confirmed.")

                # Save order
                order_data = {
                    "item": item_input,
                    "response": response,
                    "address": user_address
                }

                if not os.path.exists("orders.json"):
                    with open("orders.json", "w") as f:
                        json.dump([order_data], f, indent=2)
                else:
                    with open("orders.json", "r+") as f:
                        data = json.load(f)
                        data.append(order_data)
                        f.seek(0)
                        json.dump(data, f, indent=2)

                speak("Your order has been saved. Have a great day!")
            else:
                speak("Okay, order cancelled.")
    else:
        speak("Sorry, I can only help with ordering right now.")

if __name__ == "__main__":
    conversation()
