from gtts import gTTS
from playsound import playsound
from transformers import pipeline

# Load emotion classification model
emotion_classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-emotion")

def text_to_speech_with_emotion(text, filename):
    # Classify the emotion of the input text
    emotion = classify_emotion(text)

    # Modify the speech output based on the predicted emotion
    modified_text = modify_text_based_on_emotion(text, emotion)

    # Convert modified text to speech
    tts = gTTS(text=modified_text, lang='en')
    tts.save(filename)
    playsound(filename)

def classify_emotion(text):
    result = emotion_classifier(text)
    predicted_emotion = result[0]['label']
    return predicted_emotion

def modify_text_based_on_emotion(text, emotion):
    # Modify the text based on the predicted emotion
    if emotion == "joy":
        modified_text = add_joyful_tone(text)
    elif emotion == "sadness":
        modified_text = add_sad_tone(text)
    elif emotion == "anger":
        modified_text = add_angry_tone(text)
    else:
        modified_text = text  # No modification for neutral or other emotions
    return modified_text

def add_joyful_tone(text):
    # Add modifications to the text for a joyful tone
    modified_text = "Happily " + text
    return modified_text

def add_sad_tone(text):
    # Add modifications to the text for a sad tone
    modified_text = "Sadly " + text
    return modified_text

def add_angry_tone(text):
    # Add modifications to the text for an angry tone
    modified_text = "Angrily " + text
    return modified_text

# Example usage
input_text = "I am feeling great!"
output_file = "output.mp3"

text_to_speech_with_emotion(input_text, output_file)
