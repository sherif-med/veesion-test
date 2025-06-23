# Veesion Test: Repository Overview

This repository (`veesion-test`) contains solved tasks as requested  on human pose estimation and video classification. It leverages PyTorch Lightning for streamlined model training and MediaPipe for skeleton detection. The project is designed with modularity in mind, separating data handling, model definitions, training tasks, and utility scripts into distinct components.

### Key Design Principles

#### Separation of Concerns

- **`scripts/`**: Houses all executable scripts responsible for specific tasks such as data preprocessing (skeleton detection), model training, and model prediction. This separation ensures that the core logic within `src/` remains independent of how it's executed. Each script is a self-contained entry point for a particular workflow.
PS: Each script name has a suffix like `t1_` or `t2_` to indicate the respective task (only t2_ corresponds to 2 & 3).

- **`notebooks/`**: Contains Jupyter notebooks for exploratory data analysis and visualization (First task only).

- **`src/`**: The core source code directory, containing the building blocks. It's further sub-divided to organize different aspects of the machine learning pipeline.

- **`data/` (implicit)**: for videos and MediaPipe skeleton outputs. !! Data is version controlled because its custom !!

- **`models/` (implicit)**: Similar to `data/`, trained models and MediaPipe's `pose_landmarker.task` are stored in a `models/` directory, excluded from version control.

- **`checkpoints/` (implicit)**: Checkpoints from training runs are also excluded.

#### Modularity and Reusability

- **`src/datasets/`**: Contains classes (`SkeletonDataset`, `VideosDataset`, `FramePairDataset`) for loading and preprocessing different types of data. These are designed to be flexible and can be easily extended or reused across various tasks.

- **`src/models/`**: Defines the neural network architectures (`FrameEncoder`, `LSTMClassifier`, `LSTMVideoClassifier`, `TransformerVideoClassifier`). Each model is a distinct PyTorch `nn.Module`.

- **`src/tasks/`**: Encapsulates the training and validation logic for specific machine learning tasks using PyTorch Lightning. This includes `PoseClassificationTask`, `SSLFrameTask` (for self-supervised learning), and `VideoClassificationTask`. By using PyTorch Lightning, common training boilerplate is abstracted away, and tasks can be easily configured and run.

- **`src/callbacks/`**: Houses custom PyTorch Lightning callbacks, such as `PostPredictionCallback`, which provides a standardized way to handle post-prediction processing.

- **`src/utils/`** (or `src/viz_utils.py`, `src/models_setup.py`): Contains utility functions that support various parts of the project, like model downloading (`models_setup.py`) and visualization (`viz_utils.py`).

#### Dependency Management and Environment Setup

- **`pyproject.toml`**: Manages project dependencies and build configurations using `uv`. It explicitly lists all required packages, including torch and torchvision with specific index URLs for CUDA-enabled versions.

#### Clear Entry Points

The `scripts/` directory clearly defines the main entry points for different operations. For example, `t1_recurrent_model_train.py` is dedicated to training a recurrent model for pose classification, and `t2_video_classification_predict.py` is for making predictions with a trained video classification model.

### Strategy Rationale

- **PyTorch Lightning for Training**: PyTorch Lightning is chosen to reduce boilerplate code associated with training loops, validation, and testing. It handles device management, distributed training, and logging automatically, allowing developers to focus on the model and task logic.

- **Self-Supervised Learning (SSL) for Frame Encoding**: The `t2_frame_encoder_train.py` script and `SSLFrameTask` demonstrate a strategy of pre-training a `FrameEncoder` using self-supervised learning (specifically, the NT-Xent loss, often used in SimCLR). This allows for learning robust visual representations from video frames without requiring human-annotated labels for every frame. The pre-trained encoder can then be used as a feature extractor for downstream tasks like video classification, potentially improving performance with less labeled data.

- **Modular Model Architectures**: The models directory offers distinct model components (e.g., `LSTMClassifier`, `LSTMVideoClassifier`, `TransformerVideoClassifier`, `FrameEncoder`). This allows for easy experimentation with different architectures for sequence modeling (LSTM vs. Transformer) and feature extraction.

- **Callback System for Extensibility**: The use of PyTorch Lightning callbacks (e.g., `PostPredictionCallback`) allows for injecting custom logic into the training or prediction pipeline without modifying the core task code. This is particularly useful for tasks like saving predictions in a specific format.

- **Automated Model Download**: The `src/models_setup.py` ensures that necessary external models (like MediaPipe's pose landmarker) are automatically downloaded if not present, simplifying the setup process for new users.

## Installation and Running Instructions

To set up and run this project, follow the steps below. It is highly recommended to use a virtual environment to manage dependencies.

### Prerequisites

- Python 3.12 (as specified in `.python-version`)
- `uv` (recommended for faster dependency resolution and installation) or `pip`

### Step-by-Step Installation


#### 1. Set Up a Virtual Environment

**Using `uv` (recommended):**

```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows
```

**Alternatively, using `venv` (built-in Python module):**

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows
```

#### 2. Install Dependencies

With your virtual environment activated, install the project dependencies using `uv`:

```bash
uv sync
```

If you don't have `uv` installed, you can install it via pip and then proceed:

```bash
pip install uv
uv sync
```

This command will install all the packages listed in `pyproject.toml`, including torch and torchvision with CUDA support if your system and `uv` detect a compatible GPU.


#### 5. Prepare Data

The project expects video files and their corresponding MediaPipe skeleton JSONs in specific directories. Create the necessary data directories:

```bash
mkdir -p data/source_penn/videos
mkdir -p data/source_penn/media_pipe_skeletons
```

- Place your `.mp4` video files into `data/source_penn/videos/`
- The skeleton detection script will populate `data/source_penn/media_pipe_skeletons/` with `.json` files

## Running the Project

The `scripts/` directory contains the main executable files for different tasks.

### Task 1: Pose Classification (using skeleton data)

This task involves training a recurrent model (LSTM) to classify poses based on extracted skeleton keypoints.

#### Extract Skeletons from Videos

Before training or predicting with skeleton data, you need to extract the keypoints from your video files.

```bash
python scripts/t1_skeleton_detection.py
```

This script will read `.mp4` files from `data/source_penn/videos/`, detect human skeletons using MediaPipe, and save the keypoint data as JSON files in `data/source_penn/media_pipe_skeletons/`.

#### Train the Recurrent Pose Classification Model

To train the LSTM-based pose classifier:

```bash
python scripts/t1_recurrent_model_train.py
```

This script will:
- Load skeleton data from `MEDIA_PIPE_SKELETONS_DIR`
- Use predefined `file_list` and `labels` for training and validation (you might need to modify these lists in the script to match your dataset)
- Train an `LSTMClassifier` using PyTorch Lightning
- Save the best model checkpoint to `checkpoints/` based on `val_loss`

#### Run Prediction with the Recurrent Pose Classification Model

To make predictions using a trained pose classification model:

```bash
python scripts/t1_recurrent_model_predict.py \
    --checkpoint_path checkpoints/best-pose-classifier-epoch=XX-val_loss=X.XX.ckpt \
    --input_folder data/source_penn/media_pipe_skeletons \
    --output_folder predictions/pose_classification \
    --batch_size 4 \
    --device cpu # or cuda if you have a GPU
```

- Replace `checkpoints/best-pose-classifier-epoch=XX-val_loss=X.XX.ckpt` with the actual path to your trained model checkpoint
- The predictions will be saved to `predictions/pose_classification/predictions.txt`

### Task 2: Video Classification (using raw video frames)

This task involves training a model to classify entire videos. It supports using a pre-trained frame encoder for feature extraction and either an LSTM or Transformer for temporal modeling.

#### Train the Frame Encoder (Self-Supervised Learning)

It is highly recommended to pre-train the `FrameEncoder` using self-supervised learning (SimCLR). This helps the model learn good visual representations from frames without requiring explicit labels.

```bash
python scripts/t2_frame_encoder_train.py
```

This script will:
- Load video frames from `VIDEOS_DIR`
- Train a `FrameEncoder` using `SSLFrameTask` with the NT-Xent loss
- Save the best frame encoder checkpoint to `checkpoints/encoder/`

#### Train the Video Classification Model

To train the video classification model (which uses the pre-trained frame encoder):

```bash
python scripts/t2_video_classification_train.py
```

This script will:
- Load video data from `VIDEOS_DIR`
- Use predefined `file_list` and `labels` for training and validation (you might need to modify these lists in the script to match your dataset)
- Initialize `VideoClassificationTask` with the path to your pre-trained frame encoder (make sure `PRETRAINED_FRAME_ENCODER_PATH` in the script is correct)
- Train either an `LSTMVideoClassifier` or `TransformerVideoClassifier` based on the `TEMPORAL_MODELING` configuration in the script
- Save the best model checkpoint to `checkpoints/video_classification/`

> **Note**: The default `PRETRAINED_FRAME_ENCODER_PATH` in `t2_video_classification_train.py` points to an example checkpoint path. Ensure this matches your trained encoder.

#### Run Prediction with the Video Classification Model

To make predictions using a trained video classification model:

```bash
python scripts/t2_video_classification_predict.py \
    --checkpoint_path checkpoints/video_classification/best-video-classifier-epoch=XX-val_loss=X.XX.ckpt \
    --input_folder data/source_penn/videos \
    --output_folder predictions/video_classification \
    --batch_size 1 \
    --device cpu # or cuda if you have a GPU
```

- Replace `checkpoints/video_classification/best-video-classifier-epoch=XX-val_loss=X.XX.ckpt` with the actual path to your trained model checkpoint
- The predictions will be saved to `predictions/video_classification/predictions.txt`

## Important Notes

- **Data Paths**: Ensure that the `VIDEOS_DIR` and `MEDIA_PIPE_SKELETONS_DIR` constants in `src/__init__.py` correctly point to your data directories relative to the project root.

- **Dataset Configuration**: For training scripts (`t1_recurrent_model_train.py`, `t2_video_classification_train.py`), you will likely need to adjust the `file_list` and `labels` within the scripts to match your specific dataset.

- **GPU Usage**: The training and prediction scripts are configured to use `accelerator='auto'` (PyTorch Lightning), which will automatically detect and use a GPU if available. You can explicitly set `--device cuda` or `--device cpu` for prediction scripts.

- **Checkpoints and Logs**: Trained model checkpoints will be saved in the `checkpoints/` directory, and TensorBoard logs (if `logger=True` in Trainer) will be in `lightning_logs/`. These directories are ignored by Git. You can view TensorBoard logs by running `tensorboard --logdir lightning_logs` in your terminal and opening the provided URL in your browser.

- **Batch Size**: Adjust `BATCH_SIZE` in the configuration sections of the training and prediction scripts based on your available memory (especially GPU memory).

By following these instructions, you should be able to successfully set up, train, and run the models within the `veesion-test-main` repository.