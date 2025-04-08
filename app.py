import os
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import time
from model_module import *
from utils import *

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi'}

def allowed_file(filename, file_type):
    """Check if the uploaded file has an allowed extension"""
    if file_type == 'image':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/detect/image', methods=['POST'])
def detect_image():
    """API endpoint for image deepfake detection"""
    # Check if file is present in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename, 'image'):
        return jsonify({'error': 'File type not allowed. Please upload jpg, jpeg, or png'}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    model = VITClassifier(model_name_or_path, 2)
    model.load_state_dict(torch.load("static/saved_models/vit_deep_fake_model_v5.pth"))
    output = detect_deepfake_image(model, file_path, transform_deepfake_infer, get_device())

    
    # This would be replaced with actual model prediction
    result = output["prediction"]
    confidence = output["confidence"]
    ethical_score = output["ethical_score"] # Only for fake results
    detection_id = str(int(time.time() * 1000))  # Unique ID based on current timestamp
    file_name = filename
    file_type = 'image'
    
    # Return the result
    response = {
        'result': result,
        'confidence': confidence,
        'detection_id': detection_id,
        'file_name': file_name,
        'file_type': file_type
    }
    
    if ethical_score is not None:
        response['ethical_score'] = ethical_score
    
    # Cleanup - remove the uploaded file
    if os.path.exists(file_path):
        os.remove(file_path)
    
    return jsonify(response)

@app.route('/detect/video', methods=['POST'])
def detect_video():
    """API endpoint for video deepfake detection"""
    # Check if file is present in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename, 'video'):
        return jsonify({'error': 'File type not allowed. Please upload mp4, mov, or avi'}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    model = VITClassifier(model_name_or_path, 2)
    model.load_state_dict(torch.load("static/saved_models/vit_deep_fake_model_v5.pth"))
    output = detect_deepfake_video(model, file_path, transform_deepfake_infer, get_device())

    
    # This would be replaced with actual model prediction
    result = output["prediction"]
    confidence = output["confidence"]
    ethical_score = output["ethical_score"] # Only for fake results
    detection_id = str(int(time.time() * 1000))  # Unique ID based on current timestamp
    file_name = filename
    file_type = 'video'
    
    # Return the result
    response = {
        'result': result,
        'confidence': confidence,
        'detection_id': detection_id,
        'file_name': file_name,
        'file_type': file_type
    }
    
    if ethical_score is not None:
        response['ethical_score'] = ethical_score
    
    # Cleanup - remove the uploaded file
    if os.path.exists(file_path):
        os.remove(file_path)

    return jsonify(response)


@app.route('/api/deepfake-reasons', methods=['GET'])
def get_deepfake_reasons():
    """API endpoint to get all predefined reasons for deepfakes"""
    return jsonify(DEEPFAKE_REASONS)

@app.route('/api/submit-feedback', methods=['POST'])
def submit_deepfake_feedback():
    """API endpoint to submit user feedback for a detected deepfake"""
    data = request.json
    
    # Validate required fields
    required_fields = ['detection_id', 'reason_id', 'ethical_score', 'file_type', 'confidence_score']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Validate score range
    if not 1 <= data['ethical_score'] <= 10:
        return jsonify({'error': 'Ethical score must be between 1 and 10'}), 400
    
    # Find the reason text by ID
    reason_text = "Unknown reason"
    for reason in DEEPFAKE_REASONS:
        if reason['id'] == data['reason_id']:
            reason_text = reason['text']
            break
    
    
    # Create a new feedback record
    feedback = DeepfakeFeedback(
        is_fake=True,
        detection_id=data['detection_id'] if data.get('detection_id') else None,
        file_name=data['file_name'] if data.get('file_name') else None,
        confidence_score=data['confidence_score'] if data.get('confidence_score') else None,
        reason_id=data['reason_id'] if data.get('reason_id') else None,
        ethical_score=data['ethical_score'] if data.get('ethical_score') else None,
        file_type=data['file_type'] if data.get('file_type') else None,
        reason_text = reason_text
    )

    feedback_data = feedback.get_feedback()
    save_feedback(feedback_data)

    return jsonify({'feedback': feedback_data}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
