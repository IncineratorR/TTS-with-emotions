# TTS-with-emotions
A text to speech model with emotions
In the code above, we load an emotion classification model using the Hugging Face transformers library. The emotion_classifier pipeline is responsible for predicting the emotion associated with the input text. We define a set of modification functions (add_joyful_tone, add_sad_tone, add_angry_tone) that modify the input text based on the predicted emotion. Finally, we pass the modified text to the gTTS library to convert it into speech and play the resulting audio using the playsound library.
