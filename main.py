import os
import datetime
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import torch

# Cohere and YOLO
import cohere
from ultralytics import YOLO

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="GPT-Based Multimodal Bot", page_icon="🧠", layout="centered")

# Check API key
if not COHERE_API_KEY:
    st.error("❌ COHERE_API_KEY not found. Set it in your .env file.")
    st.stop()

# Load Cohere client
co = cohere.Client(COHERE_API_KEY)

# Load YOLOv8 model
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")  # change to 'yolov8s.pt' for better accuracy

yolo_model = load_yolo()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧠 GPT-based Multimodal Chatbot (Image + Text)")

# Upload image
image_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
caption = ""
detected_objects = []

if image_file:
    st.image(image_file, caption="Uploaded Image", use_container_width=True)

    # Open image
    image = Image.open(image_file).convert('RGB')

    # Detect objects with YOLO
    with st.spinner("🔍 Detecting objects in image..."):
        results = yolo_model.predict(image, conf=0.4, verbose=False)
        labels = results[0].names
        detected = results[0].boxes.cls.tolist()
        detected_objects = list(set([labels[int(cls_id)] for cls_id in detected]))

    # Generate caption using Cohere
    with st.spinner("📝 Generating caption using GPT..."):
        prompt = (
            f"Describe an image that contains the following objects: {', '.join(detected_objects)}. "
            "Generate a detailed and imaginative caption suitable for a chatbot."
        )

        try:
            response = co.generate(
                prompt=prompt,
                model="command-r-plus-08-2024",  # ✅ updated model
                temperature=0.7,
                max_tokens=100
            )
            caption = response.generations[0].text.strip()
            st.success("Caption generated:")
            st.markdown(f"> _{caption}_")
        except Exception as e:
            st.error(f"❌ Error generating caption: {e}")
            caption = ""

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_prompt = st.chat_input("Ask a question...")
if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Combine user input with image context
    image_context = ""
    if image_file and caption:
        image_context += f"Image caption: {caption}\n"
        if detected_objects:
            image_context += f"Objects detected: {', '.join(detected_objects)}\n"

    combined_prompt = f"{image_context}\nUser question: {user_prompt}" if image_context else user_prompt

    try:
        with st.spinner("🤖 Thinking..."):
            response = co.chat(
                model="command-r-plus-08-2024",  # ✅ updated model
                message=combined_prompt,         # ✅ correct argument
                temperature=0.7,
                max_tokens=1024,
                chat_history=[
                    {"role": "USER" if m["role"] == "user" else "CHATBOT", "message": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
            )
            reply = response.text.strip()
            st.chat_message("assistant").markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.chat_message("assistant").error(f"⚠️ Error: {e}")
