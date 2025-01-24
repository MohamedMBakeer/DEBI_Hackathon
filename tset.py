from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10)

known_faces = []  # Load your known faces
known_names = []  # Load corresponding names

# Preprocess function (same as original)
def preprocess_face(face_image):
    face_image = cv2.resize(face_image, (128, 128))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = face_image / 255.0
    return face_image

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Extract image data from the POST request
    data = request.json.get('image')
    if not data:
        return jsonify({"error": "No image provided"}), 400

    # Decode the base64-encoded image
    image_data = base64.b64decode(data)
    image = np.array(Image.open(BytesIO(image_data)))
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process the frame for face detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    response_data = []

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))

            face_image = frame[y1:y2, x1:x2]

            # Preprocess the face for comparison
            if face_image.shape[0] > 0 and face_image.shape[1] > 0:
                face_to_check = preprocess_face(face_image)
                min_distance = float('inf')
                best_match_index = -1

                for i, known_face in enumerate(known_faces):
                    distance = np.mean((known_face - face_to_check) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        best_match_index = i

                threshold = 0.3
                name = "Unknown"
                if min_distance < threshold:
                    name = known_names[best_match_index]

                response_data.append({
                    "name": name,
                    "bbox": [x1, y1, x2, y2]
                })

    return jsonify({"faces": response_data})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
