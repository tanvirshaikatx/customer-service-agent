import speech_recognition as sr
import pyttsx3
import json
import logging
import re
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# --- Setup ---
logging.basicConfig(level=logging.INFO, filename="agent.log", filemode="a", format='%(asctime)s - %(message)s')

llm = OllamaLLM(model="llama3", temperature=0.7)
embeddings = OllamaEmbeddings(model="llama3")

try:
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    logging.error(f"Failed to load vectorstore: {e}")
    exit(1)

# --- Audio Setup ---
engine = pyttsx3.init()
recognizer = sr.Recognizer()
recognizer.energy_threshold = 1000
recognizer.pause_threshold = 1.5

def speak(text):
    print(f"🤖: {text}")
    engine.say(text)
    engine.runAndWait()
    logging.info(f"Agent: {text}")

def listen():
    try:
        with sr.Microphone() as source:
            print("🎤 Listening... (Press Ctrl+C to exit)")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        text = recognizer.recognize_google(audio).lower()
        print(f"🔊 You said: {text}")
        logging.info(f"User: {text}")
        return text
    except Exception as e:
        logging.warning(f"Speech error: {e}")
        return ""

def extract_price(text):
    match = re.search(r"\$(\d+(\.\d{1,2})?)", text)
    return float(match.group(1)) if match else None

def find_menu_item(user_input):
    docs = vectorstore.similarity_search(user_input, k=3)
    for doc in docs:
        if any(word in doc.page_content.lower() for word in user_input.split()):
            return doc.page_content
    return None

# --- Main Flow ---
def chat():
    speak("Hello! Welcome to our restaurant. What would you like to order today?")
    order = []
    total = 0.0
    asked_addon = False

    while True:
        user_input = listen()
        if not user_input:
            continue

        item_info = find_menu_item(user_input)
        if item_info:
            price = extract_price(item_info)
            if price:
                item_name = item_info.split("\n")[0].strip()
                order.append((item_name, price))
                total += price
                speak(f"{item_name} has been added to your order for ${price:.2f}.")
                if not asked_addon:
                    speak("Would you like to add anything else?")
                    asked_addon = True
                continue
            else:
                speak("I found the item, but couldn't get the price. Can you try another?")
        else:
            speak("Sorry, that item is not on our menu. Please try something else.")
            continue

        if asked_addon:
            if any(word in user_input for word in ["no", "nothing", "that's it", "done", "stop"]):
                break

    if not order:
        speak("You haven't ordered anything. Exiting.")
        return

    speak(f"Your total is ${total:.2f}. Do you want to confirm your order?")
    confirm = listen()
    if "yes" not in confirm:
        speak("Order not confirmed. Thank you!")
        return

    speak("Great! Can I have your name, please?")
    name = listen()

    speak("Thanks! Now, please tell me your address.")
    address = listen()

    order_data = {
        "name": name,
        "address": address,
        "order": order,
        "total": total
    }

    with open("order.json", "w") as f:
        json.dump(order_data, f, indent=2)

    logging.info(f"Order Saved: {order_data}")
    speak("Thank you for your order! Have a great day.")

if __name__ == "__main__":
    chat()
