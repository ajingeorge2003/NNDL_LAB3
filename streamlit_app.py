import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🧠 CIFAR-10 Image Classifier Demo")
st.write("Upload an image (32x32) — the model will predict one of the 10 classes.")

# Load model (uploaded from Colab)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("./cifar10_cnn_model.h5")
    return model

model = load_model()

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")

    # Preprocess
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    st.markdown(f"### 🏷️ Predicted Class: **{CLASS_NAMES[predicted_class]}**")
    st.markdown(f"### 🔥 Confidence: **{confidence:.2f}%**")

    # Show probability bar chart
    st.bar_chart(data=dict(zip(CLASS_NAMES, predictions[0])))

st.markdown("---")
st.markdown("**Model trained on CIFAR-10 dataset using CNN and backpropagation**")