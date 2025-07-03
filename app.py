import streamlit as st
from PIL import Image
from ultralytics import YOLO
import json
import time
import cv2
import numpy as np

# Load metadata (if needed in future extensions)
try:
    with open('treat.json', 'r') as file:
        disease_data = json.load(file)
except FileNotFoundError:
    disease_data = {}

# Load YOLO model
model = YOLO('yolov8n.pt')  # Replace with your trained weights if needed

# ----------------- Home Page -----------------
def home():
    st.title("🌾 AI Powered Pest Management System")
    st.subheader("Presented By: Sukanya Varpe, Savita Giri, Rohan Gadekar")
    st.write("Jaihind College of Engineering")

    st.markdown("---")
    st.image('./static/l.jpg', caption="Precision Farming with AI", use_column_width=True)
    
    st.markdown("""
    ## How It Works
    - Upload farm images to detect insects.
    - Uses YOLO object detection to find pests.
    - Aims to reduce pesticide use through targeted interventions.
    """)

    st.markdown("## Benefits")
    st.markdown("""
    - 🟢 **Early Detection**: Prevents widespread effects.
    - 🟢 **Remote Accessibility**: Use from anywhere.
    - 🟢 **Cost-effective**: Reduces manual inspection.
    - 🟢 **Community Support**: Share insights and learn together.
    """)

    st.image('./static/h.jpeg', caption="Automated Pest Detection", use_column_width=True)
    st.info("Go to 'Detection' in the sidebar to start detecting insects on your farm images.")

# ----------------- Detection Page -----------------
def detection():
    st.title("🐛 Pest Detection")
    st.subheader("Upload an image to detect insects.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Pests"):
            st.info("Detecting pests, please wait...")

            results = model(image)

            annotated_frame = results[0].plot()
            st.image(annotated_frame, caption="Detection Results", use_column_width=True)

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.success(f"{len(boxes)} pest(s) detected!")
                for i, box in enumerate(boxes):
                    coords = box.xyxy[0].tolist()
                    st.write(f"**Pest {i+1} Coordinates:** {coords}")
            else:
                st.warning("No pests detected. Try another image with visible insects.")

# ----------------- About Page -----------------
def about():
    st.title("ℹ️ About This Project")
    st.write("""
    This project demonstrates AI-based pest detection using YOLO with a Streamlit frontend
    to enable **smart farming and pest management**.
    
    It aims to assist farmers in detecting pests early and precisely to reduce pesticide usage.
    """)

    st.markdown("## Training and Results Visuals")

    images = {
        "Detection Results": "./static/results.png",
        "Training Batch": "./static/train_batch2.jpg",
        "Validation Batch": "./static/val_batch2_labels.jpg",
        "F1 Curve": "./static/F1_curve.png",
        "PR Curve": "./static/PR_curve.png",
        "Confusion Matrix": "./static/confusion_matrix.png",
    }

    for title, path in images.items():
        try:
            img = Image.open(path)
            st.image(img, caption=title, use_column_width=True)
        except FileNotFoundError:
            st.warning(f"{title} image not found.")

# ----------------- Contact Page -----------------
def contact():
    st.title("📞 Contact Us")
    st.write("Feel free to reach out for questions, collaborations, or feedback!")

    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        subject = st.text_input("Subject")
        message = st.text_area("Message")

        submitted = st.form_submit_button("Send")

        if submitted:
            if name and email and subject and message:
                st.success("Thank you for reaching out! We will get back to you soon.")
            else:
                st.error("Please fill out all fields before submitting.")

# ----------------- Main App -----------------
def main():
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Detection", "About", "Contact Us"])

    if page == "Home":
        home()
    elif page == "Detection":
        detection()
    elif page == "About":
        about()
    elif page == "Contact Us":
        contact()

if __name__ == "__main__":
    main()
