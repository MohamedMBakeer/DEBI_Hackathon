import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Mediapipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to load dataset and extract embeddings
def load_dataset_with_mediapipe(dataset_path, resize_to=(224, 224)):
    embeddings = []
    labels = []

    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path) and not person_name.startswith('.'):
            print(f"Processing folder: {person_name}")
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                if not image_name.startswith('.') and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        print(f"Processing image: {image_path}")
                        image = cv2.imread(image_path)
                        if image is None:
                            print(f"Could not load image: {image_path}")
                            continue

                        # Convert to RGB
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Detect faces
                        results = face_detection.process(rgb_image)
                        if results.detections:
                            for detection in results.detections:
                                bboxC = detection.location_data.relative_bounding_box
                                h, w, _ = image.shape
                                x, y, w_box, h_box = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                                      int(bboxC.width * w), int(bboxC.height * h))

                                # Crop and resize the face
                                face = rgb_image[y:y + h_box, x:x + w_box]
                                face_resized = cv2.resize(face, resize_to)

                                # Flatten the resized face as embedding
                                embeddings.append(face_resized.flatten())
                                labels.append(person_name)
                        else:
                            print(f"No face detected in {image_path}")
                    except Exception as e:
                        print(f"Error processing file {image_path}: {e}")

    print(f"Total embeddings: {len(embeddings)}, Total labels: {len(labels)}")
    return np.array(embeddings), np.array(labels)

# Train and save the KNN model
def train_and_save_knn(embeddings, labels, model_path="knn_model.pkl", n_neighbors=3):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="uniform")
    knn.fit(X_train, y_train)

    with open(model_path, "wb") as file:
        pickle.dump(knn, file)
    print(f"KNN model saved to {model_path}")

    accuracy = knn.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

# Real-time recognition
def real_time_recognition(knn_model_path, confidence_threshold=0.5):
    video_capture = cv2.VideoCapture(0)

    with open(knn_model_path, "rb") as file:
        knn = pickle.load(file)

    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                                      int(bboxC.width * w), int(bboxC.height * h))

                face = rgb_frame[y:y + h_box, x:x + w_box]
                face_resized = cv2.resize(face, (224, 224)).flatten()

                probabilities = knn.predict_proba([face_resized])
                max_confidence = np.max(probabilities)

                name = "Unknown"
                if max_confidence >= confidence_threshold:
                    predicted_class = np.argmax(probabilities)
                    name = knn.classes_[predicted_class]

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({max_confidence * 100:.1f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Real-Time Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Main workflow
dataset_path = "../Dataset"
embeddings, labels = load_dataset_with_mediapipe(dataset_path)
train_and_save_knn(embeddings, labels)
real_time_recognition("knn_model.pkl", confidence_threshold=0.5)
