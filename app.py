from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import uuid
import boto3
from botocore.exceptions import NoCredentialsError
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import os
import base64
import mediapipe as mp

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# AWS S3 configurations (Ensure your credentials are secure)
AWS_ACCESS_KEY = 'AKIAYSE4N7POIRSK45NL'
AWS_SECRET_KEY = 'qkA1heE3+u1F0RLYMGgJhMWjfT/opq9SNt7YI56R'
BUCKET_NAME = 'facefilter'
REGION_NAME = 'ap-south-1'

# Initialize AWS S3 client
def upload_to_s3(file_name, bucket, object_name=None):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
                      aws_secret_access_key=AWS_SECRET_KEY, region_name=REGION_NAME)
    
    if object_name is None:
        object_name = file_name

    try:
        s3.upload_file(file_name, bucket, object_name)
        print(f"Upload Successful: {object_name}")
    except FileNotFoundError:
        print("The file was not found")
    except NoCredentialsError:
        print("Credentials not available")

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Function to download the image from a URL
def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGBA")  # Convert to RGBA
    return np.array(img)  # Convert to NumPy array

# URL of the Spiderman face filter
spiderman_url = "https://i.etsystatic.com/25290379/r/il/518dac/3894948824/il_570xN.3894948824_ttzs.jpg"

# Download the Spiderman face filter image
overlay_image = download_image(spiderman_url)

# Function to apply the filter (overlay an image)
def apply_filter(frame, overlay_img):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect faces
    results = face_detection.process(rgb_frame)

    # If faces are detected
    if results.detections:
        for detection in results.detections:
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Resize overlay image to match the face size
            overlay_resized = cv2.resize(overlay_img, (w, h))

            # Calculate the position to overlay
            y1, y2 = max(0, y), min(ih, y + h)
            x1, x2 = max(0, x), min(iw, x + w)

            # Adjust the overlay if it goes beyond the frame
            overlay_y1, overlay_y2 = 0, y2 - y1
            overlay_x1, overlay_x2 = 0, x2 - x1

            # Get the alpha channel from the overlay image
            alpha_s = overlay_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            # Loop over RGB channels
            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * overlay_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] +
                                          alpha_l * frame[y1:y2, x1:x2, c])
    return frame

@app.route('/')
def serve_index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/apply-filter', methods=['POST'])
def apply_filter_route():
    # Existing functionality remains unchanged
    pass  # Keep the existing code here

@app.route('/capture-image', methods=['POST'])
def capture_image_route():
    try:
        # Get the base64-encoded image from the request body
        data = request.json['image']
        
        # Decode the base64-encoded image
        image_data = base64.b64decode(data)
        file_name = f"captured_{uuid.uuid4()}.png"
        
        # Save the image locally
        with open(file_name, 'wb') as f:
            f.write(image_data)
        
        # Upload the saved image to the S3 bucket
        upload_to_s3(file_name, BUCKET_NAME)
        
        # Optionally, delete the local file after uploading to S3
        os.remove(file_name)

        return jsonify(success=True, message="Image captured and uploaded successfully"), 200
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@socketio.on('frame')
def handle_frame(data):
    # Decode the image
    img_data = base64.b64decode(data)
    np_data = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    # Apply the filter
    frame_with_filter = apply_filter(frame, overlay_image)

    # Encode the frame back to base64
    _, buffer = cv2.imencode('.jpg', frame_with_filter)
    frame_encoded = base64.b64encode(buffer).decode('utf-8')

    # Send back the processed frame
    emit('response_frame', frame_encoded)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=8080)