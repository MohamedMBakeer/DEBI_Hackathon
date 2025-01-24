import streamlit as st
import numpy as np
import face_recognition
import pickle
from PIL import Image
import cv2

# Constants
DATASET_FILE = "./dataset.npz"
MODEL_PATH = "./knn_model.pkl"
N_NEIGHBORS = 3

# Load dataset
@st.cache_resource
def load_dataset(file_path):
    data = np.load(file_path)
    embeddings = data['embeddings']
    labels = data['labels']
    return embeddings, labels

# Train and load KNN model
@st.cache_resource
def train_knn_model(embeddings, labels, n_neighbors=3):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn.fit(embeddings, labels)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(knn, file)
    return knn

@st.cache_resource
def load_knn_model():
    if not os.path.exists(MODEL_PATH):
        embeddings, labels = load_dataset(DATASET_FILE)
        return train_knn_model(embeddings, labels, N_NEIGHBORS)
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

# Load the KNN model
knn_model = load_knn_model()

# Streamlit UI
st.title("Real-Time Face Recognition")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Convert uploaded image to numpy array
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # Detect faces and recognize them
    rgb_image = image_np[:, :, ::-1]  # Convert to RGB for face_recognition
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = knn_model.predict([face_encoding])[0]
        cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_np, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with bounding boxes
    st.image(image_np, caption="Recognized Faces")

