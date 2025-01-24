import streamlit as st
import cv2
import pickle
import numpy as np
from PIL import Image
import face_recognition

# Load the KNN model
def load_knn_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Process image and get face embeddings
def process_image(image, knn_model, confidence_threshold=0.98):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        probabilities = knn_model.predict_proba([face_encoding])
        max_confidence = np.max(probabilities)

        if max_confidence >= confidence_threshold:
            predicted_class = np.argmax(probabilities)
            name = knn_model.classes_[predicted_class]

        results.append({"name": name, "confidence": max_confidence, "box": (top, right, bottom, left)})
    
    return results

# Streamlit UI
def main():
    st.title("Real-Time Face Recognition")
    st.sidebar.title("Settings")

    model_path = st.sidebar.text_input("KNN Model Path", "knn_model.pkl")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.98)
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    knn_model = load_knn_model(model_path)

    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        results = process_image(image, knn_model, confidence_threshold)

        for result in results:
            top, right, bottom, left = result["box"]
            name = result["name"]
            confidence = int(result["confidence"] * 100)

            # Draw bounding boxes and labels
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, f"{name} ({confidence}%)", (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        st.image(image, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
