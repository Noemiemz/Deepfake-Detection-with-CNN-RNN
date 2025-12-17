# Deepfake Detection System
CDOF3 - Security Use Cases
@2025


## Project Overview

This project implements a deep learning system for **deepfake video detection** using a **hybrid CNN–LSTM architecture**.  
The system focuses on facial regions extracted from videos and learns both:

- **spatial features** from individual frames (CNN),
- **temporal patterns** across frame sequences (LSTM).

The complete pipeline includes:
- face extraction from videos,
- data preprocessing,
- model training and validation,
- evaluation,
- inference on new, unseen videos.

---

## Dataset

### Dataset Used

We use the **FaceForensics++ dataset**, available on **Kaggle**.

This dataset contains:
- real (original) videos,
- manipulated videos created using different deepfake techniques,
- metadata describing the videos.

### Dataset Download

1. Create a Kaggle account if you do not already have one.
2. Install and configure the Kaggle API.
3. Download the dataset from Kaggle:

**FaceForensics++ (Forensic++) Dataset** (link: https://www.kaggle.com/datasets/xdxd003/ff-c23)

**or**

You can download the dataset by running the following command in your terminal:
```bash
kaggle datasets download xdxd003/ff-c23
```

## Preprocessing and Dataset Preparation

### Face Extraction

Before training the model, videos are converted into face images.

We use **YOLOv8 Face Detection** to automatically detect and extract faces from video frames.

The face extraction process:
- reads videos frame by frame,
- detects faces using YOLOv8,
- keeps the largest detected face when multiple faces are present,
- saves extracted faces into folders organized by video.

Example command for original videos:

```bash
python extract_faces.py \
  --input_folder data/videos/original \
  --output_images_dir data/images \
  --frames_per_second 2
```

Example command for manipulated videos:

```bash
python extract_faces.py \
  --input_folder data/videos/FaceSwap \
  --output_images_dir data/images_successive \
  --n_first_faces 15
```

### Image Preprocessing

Each extracted face image is processed using the following steps:

- resized to 224 × 224 pixels,

- converted into a PyTorch tensor,

- normalized using ImageNet mean and standard deviation.

These steps ensure compatibility with the ResNet50 backbone used in the CNN.


### Sequence Preparation

Videos contain different numbers of frames. To ensure consistent input sizes:

- short sequences are padded,

- long sequences are truncated.

Each video is represented by a sequence of 8 frames.


### Train / Validation Split

The dataset is split into:

- 80% training data

- 20% validation data

The split is done randomly with a fixed seed to ensure reproducibility.


## Model Training

Training is performed in the notebook:
```bash
train_model.ipynb
```

### Training Configuration

- Model: CNN–LSTM (ResNet50 + LSTM)

- Loss function: Binary Cross-Entropy

- Optimizer: Adam

- Batch size: 4 videos

- Number of epochs: 10

- Device: GPU (if available) or CPU

After training, the model weights are saved to:
- deepfake_detector.pth


## Evaluation Strategy

The model is evaluated on the validation set using:

- Precision

- Recall

- F1-score

- PR-AUC

- Confusion Matrix


## Inference on New Videos

To run inference on a new video, use:
```bash
python predict_video.py --video_path path/to/video.mp4
```

The inference pipeline performs the following steps:

- Face extraction using YOLOv8,

- Image preprocessing (resize and normalization),

- Frame grouping into a sequence,

- Loading of trained model weights,

- Final prediction.

*Prediction rule*:

- score < 0.5 → REAL

- score ≥ 0.5 → FAKE




## Tools and Libraries
To have all of the tools needed to run the project, run:
```bash
pip install -r requirements.txt
```

The project was developed in Python using the following tools:

- PyTorch and Torchvision for model building and training,

- YOLOv8 (Ultralytics) for face detection,

- OpenCV for video processing,

- Scikit-learn for evaluation metrics,

- Matplotlib and Seaborn for visualization,

- Pandas for data handling,

- TQDM for progress tracking,

- Git for version control and collaboration.


## Collaborators

Lorrain MORLET,
Aymeric MARTIN,
Auriane MARCELINO,
Augustin GALL,
Iliana JAUNAY,
Maxence VAN LAERE,
Noémie MAZEPA
