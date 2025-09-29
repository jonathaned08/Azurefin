import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import torch

# Cohere and Transformers
import cohere
from transformers import BlipProcessor, BlipForConditionalGeneration

# YOLOv8 for object detection
from ultralytics import YOLO

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Streamlit config
st.set_page_config(page_title="Multimodel Cohere Chatbot", page_icon="🤖", layout="centered")

# Check API key
if not COHERE_API_KEY:
    st.error("❌ COHERE_API_KEY not found. Set it in your .env file.")
    st.stop()

# Load Cohere client
co = cohere.Client(COHERE_API_KEY)

# Load BLIP image captioning model
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load YOLOv8
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

blip_processor, blip_model = load_blip()
yolo_model = load_yolo()

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_context" not in st.session_state:
    st.session_state.image_context = ""

st.title("🧠 COHERE CHATBOT")

# Upload image
image_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
caption = ""
detected_objects = []

if image_file:
    st.image(image_file, caption="Uploaded Image", use_container_width=True)

    # Open image
    image = Image.open(image_file).convert('RGB')

    # Image captioning (BLIP)
    with st.spinner("📝 Generating image caption..."):
        inputs = blip_processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        st.markdown(f"🖼️ Original caption: _{caption}_")

    # Enhance caption using Cohere
    with st.spinner("✨ Enhancing caption..."):
        try:
            prompt = f"Improve this image caption to be more descriptive and vivid: '{caption}'"
            enhanced = co.chat(model="command-r-plus", message=prompt)
            enhanced_caption = enhanced.text.strip()
            caption = enhanced_caption
            st.success("Enhanced caption:")
            st.markdown(f"> _{caption}_")
        except Exception as e:
            st.warning(f"Failed to enhance caption: {e}")

    # Object detection
    with st.spinner("🔍 Detecting objects..."):
        results = yolo_model.predict(image, conf=0.4, verbose=False)
        labels = results[0].names
        detected = results[0].boxes.cls.tolist()
        detected_objects = list(set([labels[int(cls_id)] for cls_id in detected]))
        if detected_objects:
            st.success("Objects detected:")
            st.markdown(f"> **{', '.join(detected_objects)}**")
        else:
            st.warning("No confident objects detected.")

    # Save to multimodal memory
    context = f"Image caption: {caption}\n"
    if detected_objects:
        context += f"Detected objects: {', '.join(detected_objects)}\n"
    st.session_state.image_context = context

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_prompt = st.chat_input("Ask a question...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Combine image + user context
    combined_prompt = f"{st.session_state.image_context}\nUser question: {user_prompt}" if st.session_state.image_context else user_prompt

    # Chat with Cohere
    try:
        with st.spinner("🤖 Thinking..."):
            response = co.chat(
                model="command-r-plus",
                message=combined_prompt,
                temperature=0.7,
                max_tokens=1024,
                chat_history=[
                    {"role": "User" if m["role"] == "user" else "Chatbot", "message": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
            )
            reply = response.text
            st.chat_message("assistant").markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.chat_message("assistant").error(f"⚠️ Error: {e}")
