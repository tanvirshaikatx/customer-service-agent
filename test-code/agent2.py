import speech_recognition as sr
import pyttsx3
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import logging
import os

# --- Logger Setup ---
logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Initialize Ollama ---
print("🤖 Initializing AI agent...")
llm = Ollama(model="llama3.2", temperature=0.7)
embeddings = OllamaEmbeddings(model="llama3.2")

# --- Load knowledge base ---
try:
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    logging.error(f"Failed to load vectorstore: {e}")
    print(f"❌ Failed to load vectorstore: {e}")
    exit(1)

# --- Speech Setup ---
engine = pyttsx3.init()
recognizer = sr.Recognizer()
recognizer.energy_threshold = 4000  # Adjust if needed
recognizer.pause_threshold = 0.8

def speak(text):
    print(f"🤖: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    try:
        with sr.Microphone() as source:
            print("🎙️ Listening... (Press Ctrl+C to exit)")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        user_text = recognizer.recognize_google(audio).lower()
        print(f"🗣️ You said: {user_text}")
        logging.info(f"User: {user_text}")
        return user_text
    except sr.WaitTimeoutError:
        print("⏳ Listening timed out.")
        return ""
    except sr.UnknownValueError:
        print("🤷 Couldn't understand audio.")
        return ""
    except Exception as e:
        logging.error(f"Speech recognition error: {e}")
        print(f"⚠️ Listening error: {e}")
        return ""

# --- Prompt Template ---
prompt_template = """
You are an expert restaurant assistant. Answer concisely and helpfully.

Relevant Information:
{context}

Conversation History:
{history}

Current Question: {input}

Assistant Response:"""
prompt = PromptTemplate.from_template(prompt_template)

# --- Chat Loop ---
def chat():
    history = []
    speak("Hello! Welcome to our restaurant. How may I help you today?")
    
    while True:
        user_input = listen()
        if not user_input:
            continue
            
        if any(x in user_input for x in ["bye", "exit", "thank you"]):
            speak("Thank you for visiting! Have a great day!")
            break
        
        try:
            # Retrieve relevant context
            docs = vectorstore.similarity_search(user_input, k=2)
            context = "\n".join([d.page_content for d in docs])

            # Generate AI response
            response = llm.invoke(prompt.format(
                context=context,
                history="\n".join(history[-4:]),
                input=user_input
            ))
            response = response.strip()

            # Speak and log
            speak(response)
            logging.info(f"AI: {response}")
            history.append(f"User: {user_input}\nAI: {response}")
        except Exception as e:
            logging.error(f"AI processing error: {e}")
            print(f"⚠️ AI Error: {e}")
            speak("I'm having trouble processing that. Please try again.")

if __name__ == "__main__":
    chat()
