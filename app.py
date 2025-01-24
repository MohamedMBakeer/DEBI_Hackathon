import os
import cv2
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp
from PIL import Image

# Constants
DATASET_FILE = "./dataset.npz"  # Path to the dataset file in the repository
RESIZE_TO = (224, 224)  # Resize dimension for training images

# Mediapipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to load dataset
def load_dataset(file_path, resize_to=(224, 224)):
    embeddings = []
    labels = []

    data = np.load(file_path)
    image_paths = data["image_paths"]
    labels_list = data["labels"]

    for image_path, label in zip(image_paths, labels_list):
        try:
            image = cv2.imread(image_path)
            if image is None:
                continue

            resized_image = cv2.resize(image, resize_to)
            embeddings.append(resized_image.flatten())
            labels.append(label)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return np.array(embeddings), np.array(labels)

# Function to train a KNN model
def train_knn(embeddings, labels, n_neighbors=3):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="uniform")
    knn.fit(X_train, y_train)

    accuracy = knn.score(X_test, y_test)
    return knn, accuracy

# Function to perform real-time face detection and recognition
def real_time_recognition(knn, confidence_threshold=0.5):
    video_capture = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = video_capture.read()
            if not ret:
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

            # Display video in Streamlit
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video_capture.release()

# Streamlit UI
st.title("Real-Time Face Recognition with KNN and Mediapipe")
st.write("This app loads a dataset, trains a KNN model, and performs real-time face detection and recognition.")

# Load dataset
if os.path.exists(DATASET_FILE):
    embeddings, labels = load_dataset(DATASET_FILE, resize_to=RESIZE_TO)
    st.write(f"Loaded {len(embeddings)} embeddings from the dataset.")

    # Train the model
    st.write("Training the KNN model...")
    knn_model, accuracy = train_knn(embeddings, labels)
    st.success(f"Model trained with {accuracy * 100:.2f}% accuracy.")

    # Start real-time recognition
    if st.button("Start Webcam"):
        st.write("Starting webcam for real-time recognition. Press 'Q' to stop.")
        real_time_recognition(knn_model)
else:
    st.error(f"Dataset file not found: {DATASET_FILE}. Please ensure it's in the repository.")
