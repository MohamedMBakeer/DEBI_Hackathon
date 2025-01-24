import os
import json
import base64
import pickle
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

app = Flask(__name__)

# Constants
DATASET_PATH = "../Dataset"
MODEL_PATH = "knn_model.pkl"
N_NEIGHBORS = 3  # Hardcoded KNN neighbors


# Function to load dataset and generate embeddings
def load_dataset(dataset_path):
    embeddings = []
    labels = []

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path) and not person_name.startswith('.'):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                if not image_name.startswith('.') and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            embeddings.append(encodings[0])
                            labels.append(person_name)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

    return np.array(embeddings), np.array(labels)


# Function to train and save the KNN model
def train_knn_model(dataset_path, model_path, n_neighbors):
    embeddings, labels = load_dataset(dataset_path)
    if len(embeddings) == 0 or len(labels) == 0:
        raise ValueError("No data found in the dataset!")

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn.fit(embeddings, labels)

    with open(model_path, "wb") as file:
        pickle.dump(knn, file)

    print(f"KNN model trained and saved to {model_path}")


# API Endpoint for real-time video recognition
@app.route('/recognize', methods=['POST'])
def recognize():
    # Check if the model exists, if not, train it
    if not os.path.exists(MODEL_PATH):
        try:
            train_knn_model(DATASET_PATH, MODEL_PATH, N_NEIGHBORS)
        except Exception as e:
            return jsonify({"error": f"Failed to train KNN model: {str(e)}"}), 500

    # Load the trained model
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
