import os
import pandas as pd
import cv2
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from ultralytics import YOLO
from supervision import Detections
import argparse
import os

def extract_faces_from_videos(
    input_folder: str,
    output_images_dir: str,
    frames_per_second: int = 1,
):
    """
    Extracts faces from all .mp4 files in the input folder and saves face crops and YOLO annotations.

    Args:
        input_folder: Path to folder containing .mp4 files.
        output_images_dir: Path to save extracted face images.
        output_annotations_dir: Path to save YOLO annotation files.
        frames_per_second: Number of frames to extract per second (default: 1).
    """
    print(f"Downloading YOLO face detection model...")
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    print(f"Model downloaded to {model_path}")
    model = YOLO(model_path,verbose=False)
    os.environ["YOLO_VERBOSE"] = "False"
    print(f"Creating output directories if they don't exist...")
    os.makedirs(output_images_dir, exist_ok=True)
    
    print(f"Processing videos in {input_folder}...")
    for video_file in tqdm(list(Path(input_folder).rglob("*.mp4")), desc="Videos"):
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps // frames_per_second) if frames_per_second > 0 else 1
        # print(f"Video FPS: {fps}, extracting every {frame_interval} frames.")
        frame_id = 0
        saved_frame_id = 0
        imgs_of_vid_dir = os.path.join(output_images_dir, video_file.stem)
        os.makedirs(imgs_of_vid_dir, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % frame_interval == 0: # process this frame
                results = model(frame)
                detections = Detections.from_ultralytics(results[0])
                if len(detections.xyxy) == 0:
                    frame_id += 1
                    continue
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
                img_name = f"frame_{saved_frame_id}.jpg"
                img_path = os.path.join(imgs_of_vid_dir, img_name)
                cv2.imwrite(img_path, face_crop)
                saved_frame_id += 1
            frame_id += 1
        cap.release()
    print(f"Face extraction and annotation completed for {frames_per_second} frames per second!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from videos and save crops/annotations.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to folder containing .mp4 files.")
    parser.add_argument("--output_images_dir", type=str, required=True, help="Path to save extracted face images.")
    parser.add_argument("--frames_per_second", type=int, default=1, help="Number of frames to extract per second.")

    args = parser.parse_args()
    extract_faces_from_videos(
        input_folder=args.input_folder,
        output_images_dir=args.output_images_dir,
        frames_per_second=args.frames_per_second,
    )

