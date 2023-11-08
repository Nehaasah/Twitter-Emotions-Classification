import streamlit as st
import numpy as np
from transformers import AutoTokenizer, TFBertModel
import tensorflow as tf
from transformers import pipeline


# Define a custom object scope for loading the model
custom_objects = {
    "TFBertModel": TFBertModel,
    "AutoTokenizer": AutoTokenizer,
}

with st.spinner("Loading model..."):
    # Inside this block, the custom object scope is active
    with tf.keras.utils.custom_object_scope(custom_objects):
        # Load the model
        model = tf.keras.models.load_model("C:/Users/Dell/streamlit_project/emotion_classification_model.h5")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("emotion_tokenizer")


# Updated mapping for emotions
encoded_dict = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}

# Create a Streamlit app
st.title("Twitter Emotion Classifier ")

# Input for user text
user_input = st.text_area("Enter the tweet here to classify emotions:", value="", height=10, max_chars=280)



# Tokenize and preprocess user input
x_val = tokenizer(
    text=user_input,
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding="max_length",
    return_tensors="tf",
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True,
)

# Predict emotion when user clicks a button
if st.button("Predict Emotion of the tweet "):
    with st.spinner("Predicting..."):
        prediction = model.predict(
            {"input_ids": x_val["input_ids"], "attention_mask": x_val["attention_mask"]}
        ) * 100

        # Display the results using the updated mapping
        emotions = {emotion.capitalize(): prediction[0][value] for emotion, value in encoded_dict.items()}
        
        # Sort emotions by percentage in descending order
        sorted_emotions = {k: v for k, v in sorted(emotions.items(), key=lambda item: item[1], reverse=True)}

        for emotion, percentage in sorted_emotions.items():
            st.write(f"{emotion}: {percentage:.2f}%")