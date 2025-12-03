import os
import argparse
import cv2
import torch
import torchvision
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np


class DeepFakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove final FC layer
        self.rnn = nn.LSTM(input_size=2048, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, C, H, W)
        batch_size, seq_len = x.size(0), x.size(1)
        cnn_features = []
        for i in range(seq_len):
            features = self.cnn(x[:, i, :, :, :])  # (batch_size, 2048)
            cnn_features.append(features)
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, seq_len, 2048)
        _, (hidden, _) = self.rnn(cnn_features)
        output = self.fc(hidden[-1])
        return torch.sigmoid(output)


def extract_first_n_faces(video_path: str, n: int = 15, images_per_second: int = 1):
    """
    Extract the first n faces from a video.
    
    Args:
        video_path: Path to the video file
        n: Number of faces to extract (default: 15)
        images_per_second: Number of frames to check per second (default: 1)
    
    Returns:
        List of face crops as numpy arrays
    """
    print(f"Downloading YOLO face detection model...")
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    # Force YOLO to run on CPU to avoid CUDA compatibility issues
    model = YOLO(model_path, verbose=False)
    model.to('cpu')
    os.environ["YOLO_VERBOSE"] = "False"
    
    print(f"Extracting faces from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // images_per_second) if images_per_second > 0 else int(fps)
    
    faces = []
    frame_id = 0
    
    while len(faces) < n:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Only found {len(faces)} faces in the video (requested {n})")
            break
        
        if frame_id % frame_interval == 0:
            results = model(frame)
            detections = Detections.from_ultralytics(results[0])
            
            if len(detections.xyxy) == 0:
                frame_id += 1
                continue
            
            # If multiple faces detected, take the largest one
            if len(detections.xyxy) > 1:
                widths = []
                for box in detections.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    widths.append(x2 - x1)
                max_area_index = widths.index(max(widths))
                detections.xyxy = [detections.xyxy[max_area_index]]
            
            box = detections.xyxy[0]
            x1, y1, x2, y2 = map(int, box)
            face_crop = frame[y1:y2, x1:x2]
            
            # Convert BGR to RGB
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            faces.append(face_crop_rgb)
            
            print(f"Extracted face {len(faces)}/{n}")
        
        frame_id += 1
    
    cap.release()
    print(f"Successfully extracted {len(faces)} faces")
    return faces


def predict_video(video_path: str, model_path: str, device: str = 'cuda', plot_faces: bool = False):
    """
    Predict if a video is real or fake.
    
    Args:
        video_path: Path to the video file
        model_path: Path to the trained model weights
        device: Device to run inference on ('cuda' or 'cpu')
        plot_faces: Whether to plot the extracted faces
    
    Returns:
        float: Prediction score between 0 (real) and 1 (fake)
    """
    # Extract faces
    faces = extract_first_n_faces(video_path, n=15, images_per_second=1)
    
    if len(faces) == 0:
        raise ValueError("No faces detected in the video!")
    
    # Plot faces if requested
    if plot_faces:
        num_faces = len(faces)
        cols = 5
        rows = (num_faces + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            if i < num_faces:
                ax.imshow(faces[i])
                ax.set_title(f'Face {i+1}', fontsize=10, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Load model
    print(f"Loading model from {model_path}...")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = DeepFakeDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Transform faces
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print(f"Processing {len(faces)} faces...")
    transformed_faces = []
    for face in faces:
        pil_image = Image.fromarray(face)
        transformed = transform(pil_image)
        transformed_faces.append(transformed)
    
    # Stack into a sequence (1, seq_len, C, H, W)
    face_sequence = torch.stack(transformed_faces, dim=0).unsqueeze(0)
    
    # Run inference
    print("Running model inference...")
    with torch.no_grad():
        output = model(face_sequence.to(device))
        score = output.squeeze().item()
    
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if a video is real or fake using deepfake detection model.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--model_path", type=str, default="deepfake_detector.pth", help="Path to the trained model weights")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run inference on")
    parser.add_argument("--plot_faces", action="store_true", help="Plot the extracted faces")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit(1)
    
    try:
        score = predict_video(args.video_path, args.model_path, args.device, args.plot_faces)
        print("\n" + "="*50)
        print(f"PREDICTION RESULT")
        print("="*50)
        print(f"Video: {args.video_path}")
        print(f"Score: {score:.4f}")
        print(f"Prediction: {'FAKE' if score > 0.5 else 'REAL'}")
        print(f"Confidence: {abs(score - 0.5) * 200:.2f}%")
        print("="*50)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        exit(1)
