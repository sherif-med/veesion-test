#!/usr/bin/env python3
"""
Download MediaPipe Pose Landmarker model with configurable settings
"""

import os
import urllib.request
from pathlib import Path

TOP_DIR = Path(__file__).parent.parent
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_FILENAME = "pose_landmarker.task"
MODELS_DIR = os.path.join(TOP_DIR, "models")  # Optional: organize models in a directory

def get_model_path():
    """Get the full path to the model file - use this in other modules"""
    return Path(MODELS_DIR) / MODEL_FILENAME

def download_model(url=MODEL_URL, filename=None, models_dir=MODELS_DIR):
    """
    Download a model file

    Args:
        url: Model download URL
        filename: Output filename (defaults to MODEL_FILENAME)
        models_dir: Directory to save models (defaults to MODELS_DIR)

    Returns:
        Path: Path to downloaded file
    """
    if filename is None:
        filename = MODEL_FILENAME

    # Create models directory if it doesn't exist
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)

    file_path = models_path / filename

    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, file_path)
        print(f"Successfully downloaded {file_path}")
        print(f"File size: {file_path.stat().st_size} bytes")
        return file_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        raise

def ensure_model_exists():
    """Download model if it doesn't exist - useful for other modules"""
    model_path = get_model_path()
    if not model_path.exists():
        print(f"Model not found at {model_path}, downloading...")
        return download_model()
    return model_path
