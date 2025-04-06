import torch
import cv2
from torchvision import transforms
from PIL import Image
import random
from facenet_pytorch import MTCNN
import numpy as np
import json

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# Define image transformations with advanced augmentations
transform_deepfake_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_deepfake_video(model, video_path, transform, device):
    model.eval()
    faces = extract_faces_from_video(video_path, device=device)
    real_count = 0
    manipulated_count = 0

    for face in faces:
        image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted].item() * 100

            if predicted == 0:
                real_count += 1
            else:
                manipulated_count += 1

    ethical_score = random.uniform(30, 95)
    prediction = 'real' if real_count > manipulated_count else 'fake'
    result = {"prediction": prediction, "confidence": confidence, "ethical_score": ethical_score}
    return result

def detect_deepfake_image(model, image_path, transform, device):
    model.eval()
    faces = extract_faces_from_image(image_path, device=device)
    if not faces:
        return {"prediction": "unknown", "confidence": 0, "ethical_score": 0}

    face = faces[0]  # Assume primary face for inference
    image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted].item() * 100

    prediction = 'real' if predicted == 0 else 'fake'
    ethical_score = random.uniform(30, 95)
    result = {"prediction": prediction, "confidence": confidence, "ethical_score": ethical_score}

    return result

def extract_faces_from_image(image_path, min_face_size=100, device='cpu'):
    detector = MTCNN(keep_all=True, device=device)
    image = cv2.imread(image_path)
    if image is None:
        return []

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces, _ = detector.detect(rgb_image)

    face_crops = []
    if faces is not None:
        for box in faces:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            if (x2 - x1) >= min_face_size and (y2 - y1) >= min_face_size:
                face = image[y1:y2, x1:x2]
                if face.size > 0:
                    face_crops.append(face)

    return face_crops

def extract_faces_from_video(video_path, min_face_size=100, device='cpu', num_frames=10):
    detector = MTCNN(keep_all=True, device=device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    face_crops = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, _ = detector.detect(rgb_frame)

        if faces is None:
            continue

        for box in faces:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if (x2 - x1) >= min_face_size and (y2 - y1) >= min_face_size:
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    face_crops.append(face)

    cap.release()
    return face_crops


DEEPFAKE_REASONS = [
    {
        "id": 1,
        "text": "Political manipulation or disinformation"
    },
    {
        "id": 2,
        "text": "Celebrity impersonation without consent"
    },
    {
        "id": 3,
        "text": "Fake news or misleading content"
    },
    {
        "id": 4,
        "text": "Harassment or bullying"
    },
    {
        "id": 5,
        "text": "Non-consensual intimate content"
    },
    {
        "id": 6,
        "text": "Identity theft or fraud"
    }, 
    {
        "id": 7,
        "text": "Parody or satire"
    },
    {
        "id": 8,
        "text": "Artistic or creative expression"
    },
    {
        "id": 9,
        "text": "Educational or demonstration purposes"
    },
    {
        "id": 10,
        "text": "Other (unspecified)"
    }
]   

class DeepfakeFeedback():
    def __init__(self, detection_id, file_name, is_fake, confidence_score, reason_id, ethical_score, file_type,
                 reason_text):
        self.detection_id = detection_id
        self.file_name = file_name
        self.is_fake = is_fake
        self.confidence_score = confidence_score
        self.reason_id = reason_id
        self.ethical_score = ethical_score  
        self.file_type = file_type
        self.reason_text = reason_text

    def get_feedback(self):
        return {
            "detection_id": self.detection_id,
            "file_name": self.file_name,
            "is_fake": self.is_fake,
            "confidence_score": self.confidence_score,
            "reason_id": self.reason_id,
            "ethical_score": self.ethical_score,
            "file_type": self.file_type,
            "reason_text": self.reason_text
        }
    
def load_feedback():
    try:
        with open('feedback.json', 'r') as f:
            feedback_dict = json.load(f)
        return feedback_dict
    except FileNotFoundError:
        return []
def save_feedback(feedback_data):
    print(feedback_data)
    feedback_dict = load_feedback()

    print(feedback_data, feedback_dict)
    data = {
            "detection_id": feedback_data['detection_id'],
            "is_fake": feedback_data['is_fake'],
            "confidence_score": feedback_data['confidence_score'],
            "reason_id": feedback_data['reason_id'],
            "ethical_score": feedback_data['ethical_score'],
            "file_type": feedback_data['file_type'],
            "reason_text": feedback_data['reason_text']
        }
    with open('feedback.json', 'w') as f:
        file_name = feedback_data['file_name']
        feedback_dict[f'{file_name}'] = data
        json.dump(feedback_dict, f, indent=4)


