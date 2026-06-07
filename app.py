import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')

# Load YOLO model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
print(f"Loading YOLO model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

@app.route('/')
def index():
    """Serves the modified index HTML file directly."""
    return send_file('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives an image, checks format, runs YOLO model,
    validates that it is a teeth-related image, 
    filters out the 'Tooth' class (class 3) from the plot,
    and returns the anomaly counts and base64 encoded result image.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # Check format: strictly only png, jpg, jpeg allowed
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed_extensions:
        return jsonify({'error': 'Invalid format. Only PNG and JPG images are allowed.'}), 400
        
    try:
        # Read image bytes
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Could not decode or read the uploaded image.'}), 400
            
        # Run YOLO inference
        results = model(img)
        result = results[0]
        
        # Get bounding boxes details
        boxes = result.boxes
        classes = boxes.cls.cpu().numpy().astype(int) if len(boxes) > 0 else np.array([], dtype=int)
        
        # Count classes: 0 -> Caries, 1 -> Cavity, 2 -> Crack, 3 -> Tooth
        caries_count = int(np.sum(classes == 0))
        cavity_count = int(np.sum(classes == 1))
        crack_count = int(np.sum(classes == 2))
        tooth_count = int(np.sum(classes == 3))
        
        # Validate if it is a teeth related image
        # If there are zero teeth-related detections (no teeth, no anomalies), it's not teeth related.
        total_teeth_detections = caries_count + cavity_count + crack_count + tooth_count
        if total_teeth_detections == 0:
            return jsonify({
                'error': 'Invalid Image! The uploaded image does not appear to be a valid dental X-ray or teeth image. Please reupload.'
            }), 200
            
        # Filter detections to exclude class 3 ('Tooth') from visualization
        # as requested by the user: "plot without the teeth class"
        keep_indices = [i for i, c in enumerate(classes) if c in [0, 1, 2]]
        
        if len(keep_indices) > 0:
            # Overwrite results boxes to only include the anomalies (Caries, Cavity, Crack)
            result.boxes = result.boxes[keep_indices]
            annotated_img = result.plot()
        else:
            # If valid teeth image but no anomalies, just return the decoded image without annotations
            annotated_img = img
            
        # Encode result image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify({
            'success': True,
            'image': img_data_url,
            'caries': caries_count,
            'cavities': cavity_count,
            'cracks': crack_count
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500

if __name__ == '__main__':
    # Start the Flask web server on port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
