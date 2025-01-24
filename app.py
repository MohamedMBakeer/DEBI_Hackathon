import os
import face_recognition
import numpy as np
import cv2
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

# Constants
DATASET_FILE = "./dataset.npz"  # Path to dataset file in the repo
RESIZE_TO = (224, 224)  # Resize dimension for training images

# Function to load dataset from an .npz file
def load_dataset(file_path, resize_to=(224, 224)):
    embeddings = []
    labels = []

    data = np.load(file_path)
    image_paths = data['image_paths']
    labels_list = data['labels']

    for image_path, label in zip(image_paths, labels_list):
        try:
            image = cv2.imread(image_path)
            if image is None:
                continue

            resized_image = cv2.resize(image, resize_to)
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)

            if encodings:
                embeddings.append(encodings[0])
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

# Function to perform real-time face recognition
def real_time_recognition(knn, confidence_threshold=0.5):
    video_capture = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations, face_encodings = get_face_embeddings(rgb_frame)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            probabilities = knn.predict_proba([face_encoding])
            max_confidence = np.max(probabilities)

            if max_confidence >= confidence_threshold:
                predicted_class = np.argmax(probabilities)
                name = knn.classes_[predicted_class]

            confidence_percentage = int(max_confidence * 100)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence_percentage}%)", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display video in Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()

# Function to extract face embeddings
def get_face_embeddings(image):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_locations, face_encodings

# Streamlit UI
st.title("Real-Time Face Recognition with KNN")
st.write("This app loads a dataset, trains a KNN model, and performs real-time face recognition.")

# Load dataset from .npz
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
