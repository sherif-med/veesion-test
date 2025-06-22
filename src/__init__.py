from pathlib import Path

from .models_setup import ensure_model_exists

ensure_model_exists()

VIDEOS_DIR = Path(__file__).parent.parent / "data" / "source_penn" / "videos"
MEDIA_PIPE_SKELETONS_DIR = Path(__file__).parent.parent / "data" / "source_penn" / "media_pipe_skeletons"