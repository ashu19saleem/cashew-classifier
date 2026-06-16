import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np
import json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

st.set_page_config(page_title="Cashew Grade Classifier", page_icon="🌰")

# Initialize Firebase from Streamlit secrets
try:
    firebase_admin.get_app()
except ValueError:
    firebase_secrets = dict(st.secrets["firebase"])
    firebase_secrets["private_key"] = firebase_secrets["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(firebase_secrets)
    firebase_admin.initialize_app(cred)
db = firestore.client()

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cashew_classifier.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()
IMG_SIZE = (160, 160)

st.title("🌰 Cashew Nut Grade Classifier")
st.write("Upload a cashew image to predict its grade.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width="stretch")

    img_resized = img.resize(IMG_SIZE)
    img_array = img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    pred_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"Predicted Grade: **{pred_class}**")
    st.info(f"Confidence: {confidence:.2f}%")

    # Log prediction to Firestore
    db.collection("predictions").add({
        "prediction": pred_class,
        "confidence": float(confidence),
        "timestamp": datetime.utcnow().isoformat()
    })