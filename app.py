import streamlit as st
import cv2
import numpy as np
import pickle
import face_recognition
from PIL import Image

# Load the pre-trained KNN model
def load_knn_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Function to process the frame and perform face recognition
def process_frame(frame, knn_model, confidence_threshold):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations, face_encodings = get_face_embeddings(rgb_frame)
    annotated_frame = frame.copy()
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        probabilities = knn_model.predict_proba([face_encoding])
        max_confidence = np.max(probabilities)

        if max_confidence >= confidence_threshold:
            predicted_class = np.argmax(probabilities)
            name = knn_model.classes_[predicted_class]
        
        confidence_percentage = int(max_confidence * 100)
        
        # Annotate the frame with the name and confidence
        cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{name} ({confidence_percentage}%)", 
                    (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return annotated_frame

# Function to get face embeddings
def get_face_embeddings(image):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_locations, face_encodings

# Streamlit App
st.title("Real-Time Face Recognition with Streamlit")
st.sidebar.title("Settings")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.98, 0.01)
model_path = st.sidebar.text_input("Model Path", "knn_model.pkl")
run_button = st.sidebar.button("Start Recognition")

if model_path and run_button:
    st.text("Loading the KNN model...")
    knn_model = load_knn_model(model_path)
    st.text("Model loaded successfully!")

    # Start video feed
    st.text("Starting the webcam...")
    video_capture = cv2.VideoCapture(0)
    frame_window = st.image([])

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.warning("Failed to capture video. Check your camera.")
                break

            # Process the frame and get annotated frame
            annotated_frame = process_frame(frame, knn_model, confidence_threshold)
            
            # Convert to RGB for Streamlit display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(annotated_frame_rgb, channels="RGB")

            if st.sidebar.button("Stop Recognition"):
                break

    finally:
        video_capture.release()
        st.text("Webcam released. Recognition stopped.")
