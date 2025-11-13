import os
import pandas as pd
import cv2
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from ultralytics import YOLO
from supervision import Detections
import argparse

def extract_faces_from_videos(
    input_folder: str,
    output_images_dir: str,
    output_annotations_dir: str,
    metadata_csv: str,
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

    print(f"Creating output directories if they don't exist...")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)
    
    print(f"Loading metadata from {metadata_csv}...")
    metadata_df = pd.read_csv(metadata_csv)
    metadata_df['File Path'] = f'{input_folder}/' + metadata_df['File Path'].astype(str)
    
    print(f"Processing videos in {input_folder}...")
    for video_file in tqdm(list(Path(input_folder).rglob("*.mp4")), desc="Videos"):
        # print(f"\nProcessing video: {video_file}")
        row_match = metadata_df[metadata_df['File Path'] == video_file.as_posix()]
        if row_match.empty:
            print(f"Metadata for video {video_file} not found. Skipping.")
            continue
        label_video = int(row_match.iloc[0]['Label'])
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps // frames_per_second) if frames_per_second > 0 else 1
        # print(f"Video FPS: {fps}, extracting every {frame_interval} frames.")
        frame_id = 0
        saved_frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                # print(f"End of video {video_file}")
                break
            
            if frame_id % frame_interval == 0:
                # print(f"Processing frame {frame_id}...")
                results = model(frame)
                detections = Detections.from_ultralytics(results[0])
                # print(f"Detected {len(detections.xyxy)} faces in frame {frame_id}.")
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
                    # print(f"Multiple faces detected. Keeping only the largest face with width {max(widths)}.")
                
                box = detections.xyxy[0]
                
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[y1:y2, x1:x2]

                img_name = f"{video_file.stem}_frame_{saved_frame_id}.jpg"
                img_path = os.path.join(output_images_dir, img_name)
                cv2.imwrite(img_path, face_crop)
                # print(f"Saved face crop: {img_path}")

                img_height, img_width = face_crop.shape[:2]
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                annotation_line = f"{label_video} {x_center} {y_center} {width} {height}\n"
                annotation_path = os.path.join(output_annotations_dir, f"{video_file.stem}_frame_{saved_frame_id}.txt")
                with open(annotation_path, "w") as f:
                    f.write(annotation_line)
                # print(f"Saved annotation: {annotation_path}")
                saved_frame_id += 1

            frame_id += 1

        cap.release()
        # print(f"Finished processing video: {video_file}")

    print(f"Face extraction and annotation completed for {frames_per_second} frames per second!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from videos and save crops/annotations.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to folder containing .mp4 files.")
    parser.add_argument("--output_images_dir", type=str, required=True, help="Path to save extracted face images.")
    parser.add_argument("--output_annotations_dir", type=str, required=True, help="Path to save YOLO annotation files.")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV file.")
    parser.add_argument("--frames_per_second", type=int, default=1, help="Number of frames to extract per second.")

    args = parser.parse_args()
    extract_faces_from_videos(
        input_folder=args.input_folder,
        output_images_dir=args.output_images_dir,
        output_annotations_dir=args.output_annotations_dir,
        metadata_csv=args.metadata_csv,
        frames_per_second=args.frames_per_second,
    )


# extract_faces_from_videos(
#     videos_root="data/videos/tests",
#     images_root="data/images/tests",
#     annotations_root="data/annotations/tests",
#     metadata_csv="data/metadata.csv",
#     frames_per_second=2,
# )


# example usage:
# python src/extract_faces.py \
#    --input_folder data/videos/tests \
#    --output_images_dir data/images/tests \
#    --output_annotations_dir data/annotations/tests \
#    --frames_per_second 2
