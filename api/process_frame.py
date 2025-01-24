import numpy as np
import os
import pickle
from flask import Flask, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
import face_recognition
from PIL import Image
from io import BytesIO
import base64
import requests

app = Flask(__name__)

# Constants
DATASET_FILE = "./dataset.npz"  # Path to the .npz file
MODEL_PATH = "knn_model.pkl"  # Path to save/load the KNN model
N_NEIGHBORS = 3  # Number of neighbors for KNN
GITHUB_DATASET_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/dataset.npz"  # Update this URL


# Function to load dataset from .npz file
def load_dataset_from_npz(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    # Load the compressed .npz file
    data = np.load(file_path)
    embeddings = data['embeddings']
    labels = data['labels']

    print(f"Loaded dataset: {len(embeddings)} embeddings, {len(labels)} labels")
    return embeddings, labels


# Function to download and load dataset dynamically if not present
def download_and_load_dataset(url, file_path):
    if not os.path.exists(file_path):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(file_path, "wb") as file:
            file.write(response.content)

    return load_dataset_from_npz(file_path)


# Function to train and save the KNN model
def train_knn_model(embeddings, labels, model_path, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn.fit(embeddings, labels)

    # Save the model
    with open(model_path, "wb") as file:
        pickle.dump(knn, file)

    print(f"KNN model trained and saved to {model_path}")


# API Endpoint for real-time video recognition
@app.route('/recognize', methods=['POST'])
def recognize():
    # Ensure the dataset and model exist
    try:
        embeddings, labels = download_and_load_dataset(GITHUB_DATASET_URL, DATASET_FILE)
    except Exception as e:
        return jsonify({"error": f"Failed to load dataset: {str(e)}"}), 500

    # Check if the model exists, and train it if not
    if not os.path.exists(MODEL_PATH):
        print("Training KNN model...")
        train_knn_model(embeddings, labels, MODEL_PATH, N_NEIGHBORS)

    # Load the trained KNN model
    with open(MODEL_PATH, "rb") as file:
        knn = pickle.load(file)

    # Decode the image from the request
    data = request.json.get('image')
    if not data:
        return jsonify({"error": "No image provided!"}), 400

    try:
        # Decode base64 image and convert to numpy array
        image_data = base64.b64decode(data)
        image = np.array(Image.open(BytesIO(image_data)))
        rgb_image = image[:, :, ::-1]  # Convert to RGB for face_recognition

        # Detect faces and extract embeddings
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Recognize faces
        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            predictions = knn.predict([face_encoding])
            name = predictions[0] if len(predictions) > 0 else "Unknown"
            results.append({
                "name": name,
                "bbox": [top, right, bottom, left]
            })

        return jsonify({"faces": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
