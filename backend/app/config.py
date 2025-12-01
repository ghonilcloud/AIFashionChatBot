from pathlib import Path

# Base directory (root of project)
BASE_DIR = Path(__file__).resolve().parent.parent

# Media folders
MEDIA_DIR = BASE_DIR / "media"
UPLOADS_DIR = MEDIA_DIR / "uploads"
GENERATED_DIR = MEDIA_DIR / "generated"

# Make sure these dirs exist at runtime
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)