import torch
import cv2
from torchvision import transforms
from PIL import Image
import random
from facenet_pytorch import MTCNN
import numpy as np
import json,os,time, requests
from tqdm import tqdm

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


DEEPFAKE_REASONS_GENERAL = [
    {
        "id": 1,
        "text": "Provocative"
    },
    {
        "id": 2,
        "text": "Non Provocative"
    },
    {
        "id": 3,
        "text": "Not Sure"
    }
]

DEEPFAKE_REASONS_PERSONALITY = [
    {
        "id": 1,
        "text": "Influential"
    },
    {
        "id": 2,
        "text": "Non Influential"
    }
]

DEEPFAKE_REASONS_EMOTIONS = [
    {
        "id": 1,
        "text": "Weird Face"
    },
    {
        "id": 2,
        "text": "Angry Face"
    },
    {
        "id": 3,
        "text": "Happy Face"
    },
    {
        "id": 4,
        "text": "Sad Face"
    },
    {
        "id": 5,
        "text": "Surprised Face"
    },
    {
        "id": 6,
        "text": "Disgusted Face"
    }, 
    {
        "id": 7,
        "text": "Neutral Face"
    },
    {
        "id": 8,
        "text": "Other (unspecified)"
    }
]

DEEPFAKE_REASONS_BROAD = [
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
    def __init__(self, detection_id, file_name, is_fake, confidence_score, categories, file_type):
        self.detection_id = detection_id
        self.file_name = file_name
        self.is_fake = is_fake
        self.confidence_score = confidence_score 
        self.file_type = file_type
        self.categories = categories

    def get_feedback(self):
        return {
            "detection_id": self.detection_id,
            "file_name": self.file_name,
            "is_fake": self.is_fake,
            "confidence_score": self.confidence_score,
            "file_type": self.file_type,
            "categories": self.categories
        }
    
def load_feedback():
    try:
        with open('feedback.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def generate_unique_filename(feedback_category, file_name):
    if file_name not in feedback_category:
        return file_name

    base, ext = os.path.splitext(file_name)
    unique_file_name = f"{base}_{time.strftime('%Y%m%d%H%M%S')}{ext}"
    
    while unique_file_name in feedback_category:
        unique_file_name = f"{base}_{time.strftime('%Y%m%d%H%M%S')}{ext}"
    return unique_file_name

def save_feedback(feedback_data):
    try:
        feedback_json = json.loads(get_latest_feedback_json())
        feedback_dict = feedback_json['record']
        base_data = {
            "detection_id": feedback_data['detection_id'],
            "is_fake": feedback_data['is_fake'],
            "confidence_score": feedback_data['confidence_score'],
            "file_type": feedback_data['file_type']
        }
        
        file_name = feedback_data['file_name']
        categories = feedback_data.get('categories', {})

        for category, values in categories.items():
            data = base_data.copy()  # Copy to avoid mutable reference issues.
            data['reason_id'] = values['reason_id']
            data['reason_text'] = values['reason_text']
            data['ethical_score'] = values['ethical_score']
            
            if category not in feedback_dict:
                feedback_dict[category] = {}
            
            unique_file_name = generate_unique_filename(feedback_dict[category], file_name)
            
            feedback_dict[category][unique_file_name] = data

        update_feedback_json(feedback_dict)
        
    except Exception as e:
        print("Error while saving feedback:", e)

def get_latest_feedback_json():
    url = 'https://api.jsonbin.io/v3/b/67f690ed8960c979a5817a26/latest'
    headers = {
    'X-Master-Key': '$2a$10$LtB0NH4awp0VKCi4WIBrne7uY7C.gNd8UW6hc4ruQibQC4a0ZcwMu'
    }

    req = requests.get(url, json=None, headers=headers)
    return req.text

def update_feedback_json(updated_json):
    url = 'https://api.jsonbin.io/v3/b/67f690ed8960c979a5817a26'
    headers = {
    'Content-Type': 'application/json',
    'X-Master-Key': '$2a$10$LtB0NH4awp0VKCi4WIBrne7uY7C.gNd8UW6hc4ruQibQC4a0ZcwMu'
    }

    req = requests.put(url, json=updated_json, headers=headers)
    print(req.text)


def get_json_dictionary():
    feedback_json = json.loads(get_latest_feedback_json())
    feedback_dict = feedback_json['record']
    return feedback_dict

def extract_faces_from_image(image_path, output_folder, min_face_size=100, device='cpu'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    detector = MTCNN(keep_all=True, device=device)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Load and convert image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces, _ = detector.detect(rgb_image)

    if faces is None or len(faces) == 0:
        print("No faces detected.")
        return

    face_count = 0
    for i, box in enumerate(faces):
        x1, y1, x2, y2 = map(int, box)

        # Ensure box is within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        width, height = x2 - x1, y2 - y1
        if width < min_face_size or height < min_face_size:
            continue

        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        face_filename = os.path.join(output_folder, f"{image_name}_face{i}.jpg")
        cv2.imwrite(face_filename, face_crop)
        face_count += 1

    print(f"Done! Extracted {face_count} face(s) from {image_path} to {output_folder}")


def extract_faces_from_video(video_path, output_folder, min_face_size=100, device='cpu', num_frames=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    detector = MTCNN(keep_all=True, device=device)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract video name safely

    # **Select evenly spaced frames (5 frames across video)**
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    # **Find already processed frames**
    existing_files = {f for f in os.listdir(output_folder) if f.startswith(video_name)}
    processed_frames = {int(f.split('_frame')[1].split('_')[0]) for f in existing_files if '_frame' in f}

    with tqdm(total=len(frame_indices), desc=f"Processing {video_name}.mp4", unit="frame", dynamic_ncols=True, leave=True) as pbar:
        for frame_index in frame_indices:
            if frame_index in processed_frames:
                pbar.update(1)  # Skip already processed frames
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, _ = detector.detect(rgb_frame)

            # **Skip frame if no faces are detected**
            if faces is None or len(faces) == 0:
                pbar.update(1)
                continue  

            for i, box in enumerate(faces):
                x1, y1, x2, y2 = map(int, box)

                # **Ensure bounding box is within valid range**
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                width, height = x2 - x1, y2 - y1

                # **Skip small faces**
                if width < min_face_size or height < min_face_size:
                    continue  

                face_crop = frame[y1:y2, x1:x2]

                # **Avoid saving empty images**
                if face_crop.size == 0:
                    continue  

                face_filename = os.path.join(output_folder, f"{video_name}_frame{frame_index}_face{i}.jpg")
                cv2.imwrite(face_filename, face_crop)

            pbar.update(1)  # Update tqdm progress bar

    cap.release()
    print(f"Processing complete! Faces saved in {output_folder}")