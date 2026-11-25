"""
Download pretrained models from Google Drive.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
DRIVE_FILE_ID = "1Vn2WszmgGQqtPWZORmDE2fSfhkiTWpIq"

def main():
    if os.path.exists(BEST_MODEL_PATH) and os.path.getsize(BEST_MODEL_PATH) > 1000000:
        print(f"best_model.pth already exists at {BEST_MODEL_PATH}")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Downloading best_model.pth from Google Drive...")
    try:
        import gdown
        gdown.download(id=DRIVE_FILE_ID, output=BEST_MODEL_PATH, quiet=False)
        print("Download complete!")
    except ImportError:
        print("Please install gdown (`pip install gdown`) to download automatically, or manually download from Google Drive:")
        print(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}")

if __name__ == "__main__":
    main()
