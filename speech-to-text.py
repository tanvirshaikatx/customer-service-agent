import speech_recognition as sr

def recognize_until_silent():
    recognizer = sr.Recognizer()
    
    # Optional: tweak sensitivity
    recognizer.pause_threshold = 2.0  # seconds of silence before considering it 'done'

    with sr.Microphone() as source:
        print("🎤 Listening... Speak now (it will stop after you pause):")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        audio = recognizer.listen(source)  # records until user stops talking

    try:
        text = recognizer.recognize_google(audio)
        print("📝 You said:", text)
    except sr.UnknownValueError:
        print("🤔 Sorry, couldn't understand the audio.")
    except sr.RequestError as e:
        print(f"🚫 API request failed: {e}")

if __name__ == "__main__":
    recognize_until_silent()
