import streamlit as st
import lime.lime_text
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the trained model
model = load_model("fake_news_model.h5")  # Make sure you saved your model

optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)  # Clipping helps prevent instability
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Function to preprocess text
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=256)  # Ensure same max length as training
    return padded

# Function for LIME explanation
def model_predict(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=256)
    preds = model.predict(padded)
    return np.array([[1 - p[0], p[0]] for p in preds])  # Convert to probability format

st.title("📰 Fake News Detection with Explainability")

# User input
article = st.text_area("Enter a news article:")

if article:
    explainer = LimeTextExplainer(class_names=["Fake", "Real"])
    explanation = explainer.explain_instance(article, model_predict, num_features=10)

    # Show model prediction
    pred = model_predict([article])[0][1]  # Probability of being "Real"
    label = "REAL" if pred > 0.5 else "FAKE"
    st.write(f"**Prediction:** {label} (Confidence: {pred:.2%})")

    # Display explanation
    st.subheader("🔍 How did the model decide?")
    st.components.v1.html(explanation.as_html(), height=500, scrolling=True)
