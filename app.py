import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# Mediapipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Streamlit UI
st.title("Real-Time Face Detection with Mediapipe")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Convert uploaded image to numpy array
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # Mediapipe face detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                # Draw bounding box and landmarks
                mp_drawing.draw_detection(image_np, detection)

    # Display the image with bounding boxes
    st.image(image_np, caption="Detected Faces", use_column_width=True)
