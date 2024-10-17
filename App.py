import os
import torch
import requests
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer


file_id = '1UadmEDMtYlhUhE26gRB-INBdGj0UIvHe'
model_url = f"https://drive.google.com/uc?export=download&id={file_id}"
response = requests.get(model_url)
with open("bert_emotion_model.pth", "wb") as f:
    f.write(response.content)

# Load the saved BERT model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=28)

# Load the model state from the downloaded file
model.load_state_dict(torch.load("bert_emotion_model.pth", map_location=torch.device('cpu')))

# Set the device (GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# Load GPT-2 model and tokenizer for generating supportive responses
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.to(device)

# Streamlit UI setup
st.title("Mental Health Emotion Support")
st.write("This app provides mental health support based on the classified emotion in the text.")

# User input
user_input = st.text_area("Enter your text here:")

# Emotion classification using BERT
def classify_emotion_bert(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    return probabilities

# Generate a supportive response using GPT-2
def generate_supportive_response_gpt2(text, emotion):
    prompt = f"The detected emotion is {emotion}."
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = gpt2_model.generate(inputs, max_length=100, num_return_sequences=1)
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Emotion columns
emotion_columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                   'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                   'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                   'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                   'relief', 'remorse', 'sadness', 'surprise', 'neutral']

# Display results
if st.button("Classify Emotion"):
    if user_input:
        # BERT Classification
        bert_emotions = classify_emotion_bert(user_input)

        # Determine the most likely emotion for supportive response
        likely_emotion = emotion_columns[bert_emotions.argmax()]

        # Generate Supportive Response
        st.subheader("Supportive Response:")
        supportive_response = generate_supportive_response_gpt2(user_input, likely_emotion)
        st.write(supportive_response)
    else:
        st.write("Please enter some text to classify.")
