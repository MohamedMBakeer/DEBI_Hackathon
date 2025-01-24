import os
import cv2
import numpy as np
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import pickle
import mediapipe as mp
from PIL import Image

# Constants
MODEL_PATH = "./knn_model.pkl"  # Path to the pre-trained KNN model
RESIZE_TO = (224, 224)  # Resize dimension for training images

# Mediapipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to load a pre-trained KNN model
def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        st.success(f"Loaded pre-trained model from {model_path}")
        return model
    else:
        st.error(f"Model file not found: {model_path}")
        return None

# Function to perform real-time face detection and recognition
def real_time_recognition(knn, confidence_threshold=0.5):
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        st.error("Unable to access the webcam. Please check your device or permissions.")
        return

    frame_placeholder = st.empty()

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        st.write("Webcam started. Press 'Q' in the window to stop.")
        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Failed to grab frame from the webcam.")
                break

            # Detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = (
                        int(bboxC.xmin * iw),
                        int(bboxC.ymin * ih),
                        int(bboxC.width * iw),
                        int(bboxC.height * ih),
                    )
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Dummy face embedding for recognition (replace with real embeddings if available)
                    dummy_face_encoding = np.random.rand(128)  # Placeholder for testing
                    name = "Unknown"
                    if knn:
                        probabilities = knn.predict_proba([dummy_face_encoding])
                        max_confidence = np.max(probabilities)
                        if max_confidence >= confidence_threshold:
                            predicted_class = np.argmax(probabilities)
                            name = knn.classes_[predicted_class]

                    # Display recognition results
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display video in Streamlit
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Stop webcam on 'Q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video_capture.release()
    cv2.destroyAllWindows()

# Streamlit UI
st.title("Real-Time Face Recognition with Pre-Trained KNN Model")
st.write("This app uses a pre-trained KNN model to perform real-time face recognition.")

# Load pre-trained KNN model
knn_model = load_model(MODEL_PATH)

# Real-time recognition
if knn_model:
    if st.button("Start Webcam"):
        st.write("Initializing webcam for real-time recognition. Press 'Q' to stop.")
        real_time_recognition(knn_model)
else:
    st.warning("Please ensure the model file is available.")
