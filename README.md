# ML Fish Classifier & Eco-Dashboard

An AI-based system for **fish detection, counting, species identification, density estimation, spatial heat mapping, video tracking, GIS mapping, and water quality analysis** in Indian freshwater ecosystems.

Includes an interactive **Streamlit Web App (`app.py`)**, automatic **CUDA / Apple Silicon (MPS) / CPU** device support, and CLI tools for every module.

---

## Results Summary

| Module | Model | Metric | Value |
|---|---|---|---|
| Species Classification | EfficientNetB0 | Test Accuracy | 95.00% |
| Fish Detection | YOLOv8n | mAP@50 | 92.3% |
| Water Quality | XGBoost (Classifier + Regressor) | Potability & pH | Pretrained |
| Density Estimation | Formula-based | — | fish/m² |
| Heat Map | OpenCV Gaussian Blur | — | spatial overlay |
| Video Tracking | YOLOv8 + ByteTrack | Unique IDs | no double counting |

---

## Quick Start (Streamlit Web App)

### 1. Clone the repository
```bash
git clone https://github.com/ayush-7677/fish_detection.git
cd fish_detection
```

### 2. Create virtual environment & install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **macOS note:** XGBoost requires OpenMP (`libomp`). If you see a `libxgboost.dylib` / `libomp` error, install it with:
> ```bash
> brew install libomp
> ```

### 3. Download pretrained classification weights (`best_model.pth`)
```bash
python src/download_models.py
```
*(Automatically downloads `models/best_model.pth` from Google Drive. `models/yolo_fish/weights/best.pt` and the XGBoost water quality models are already included in `models/`.)*

### 4. Run the Streamlit Web App
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## Web App Modes (`app.py`)

| Mode | What it does |
|---|---|
| **Fish Detection & Counting (YOLOv8)** | Upload an image (or use `sample_fish.jpg`) to detect fish, draw bounding boxes, and view confidence stats. |
| **Species Classification (EfficientNetB0)** | Identifies the fish species with Top-5 predictions and confidence scores. Automatically matches 31-class (`class_names_31.json`) or 73-class (`class_names.json`) checkpoints. |
| **Density Estimation** | Estimates areal fish density (`fish/m²`) and total pond population from camera depth, FOV, and pond area. |
| **Spatial Heat Map** | Generates a spatial density heat map overlay over the pond image. |
| **Water Quality Analysis** | Takes Turbidity, Hardness, and TDS sliders to predict water potability and pH using XGBoost. |

### Detection Settings (Sidebar Sliders)
- **Confidence Threshold** (default `0.25`): Lower (`0.15–0.25`) to detect more fish in dense schools; raise (`0.40–0.60+`) to filter false positives.
- **IoU / NMS Threshold** (default `0.30`): Keep low (`0.20–0.35`) for schools of fish so nearby fish aren't merged into a single box. Uses `agnostic_nms=True`.
- **Max Detections** (default `300`, up to `500`): Maximum fish boxes allowed per image.

---

## Dataset

| Split | Images |
|---|---|
| Train | 24,373 |
| Val | 5,416 |
| Test | 3,400 |
| **Total** | **33,189** |

**73 Indian freshwater fish species**

**Sources:**
- [Kaggle - Identify the Fish](https://www.kaggle.com/code/ammarbhilwarawala/identify-the-fish/input) — 31 species
- [AS-FishFresh Augmented Dataset (Mendeley)](https://data.mendeley.com/datasets/sdhp9bmrz5/2) — 48 Assam freshwater species (merged & deduplicated via `src/merge_datasets.py`)

---

## Project Structure

```text
fish_detection/
├── app.py                          # Streamlit web dashboard (5 modes)
├── sample_fish.jpg                 # Built-in sample image for testing
├── requirements.txt
├── detection/
│   └── data.yaml                   # YOLOv8 dataset config
├── models/
│   ├── best_model.pth              # EfficientNetB0 checkpoint (via download_models.py)
│   ├── class_names.json            # 73 species list
│   ├── class_names_31.json         # 31 species list (auto-matched to 31-class checkpoint)
│   ├── model_clf.json              # XGBoost potability classifier
│   ├── model_reg.json              # XGBoost pH regressor
│   ├── confusion_matrix.png
│   └── yolo_fish/
│       └── weights/best.pt         # YOLOv8n trained fish detector
├── src/
│   ├── download_models.py          # Downloads best_model.pth from Google Drive
│   ├── merge_datasets.py           # Merges Kaggle + Assam datasets
│   ├── gis_map.py                  # Interactive Folium GIS map generator
│   ├── integrated_report.py        # All-in-one detection + density + water quality CLI
│   ├── classification/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   └── detection/
│       ├── train_yolo.py
│       ├── predict_yolo.py
│       ├── count_fish.py
│       ├── density.py
│       ├── heatmap.py
│       └── track_fish.py           # ByteTrack video tracking
└── water_quality/
    └── main.py                     # FastAPI water quality service
```

---

## CLI Usage

All scripts automatically detect **CUDA → Apple Silicon (`mps`) → CPU** and use relative paths.

### 1. Species Classification
```bash
python src/classification/predict.py --image "path/to/fish.jpg"
```

### 2. Fish Detection & Counting
```bash
python src/detection/predict_yolo.py --image "path/to/pond.jpg"
```

### 3. Detect + Identify Species
```bash
python src/detection/count_fish.py --image "path/to/pond.jpg"
```

### 4. Density Estimation
```bash
python src/detection/density.py --image "path/to/pond.jpg" --depth 2 --fov 90 --pond_area 2000
```
- `--depth` : camera depth in meters
- `--fov` : camera field of view in degrees
- `--pond_area` : total pond area in m²

### 5. Heat Map
```bash
python src/detection/heatmap.py --image "path/to/pond.jpg"
```

### 6. Fish Tracking (Video)
```bash
python src/detection/track_fish.py --video "path/to/video.mp4"
```
Uses **YOLOv8 + ByteTrack** to assign persistent IDs across frames and avoid double-counting.

### 7. GIS Mapping
```bash
python src/gis_map.py \
  --lat 23.4120 \
  --lon 85.3094 \
  --location "Kanke Dam, Ranchi" \
  --total 27 \
  --density 2.25 \
  --ph 7.59 \
  --water_safe 1 \
  --species_json '{"Catfish":3,"Goby":3,"Glass Perchlet":9}'
```
Serve the output map:
```bash
cd models/gis
python -m http.server 8080
# Open: http://localhost:8080/fish_gis_map.html
```

### 8. Integrated Report (All-in-One)
```bash
python src/integrated_report.py \
  --image "path/to/pond.jpg" \
  --turbidity 3.9 \
  --hardness 196 \
  --tds 500 \
  --depth 2 \
  --fov 90 \
  --pond_area 2000
```

### 9. Water Quality FastAPI Service
```bash
uvicorn water_quality.main:app --reload
# API docs at: http://localhost:8000/docs
```

### 10. Train from Scratch
```bash
# Classification (EfficientNetB0)
python src/classification/train.py

# Detection (YOLOv8n)
python src/detection/train_yolo.py
```

---

## Pipeline Flow

```text
Input Image / Video
   ↓
YOLOv8n → Detect fish + draw boxes + count
   ↓
EfficientNetB0 → Crop each box + identify species
   ↓
Density Module → fish/m² + estimated pond population
   ↓
Heat Map / ByteTrack / Folium GIS Map
   ↓
XGBoost → Water potability + pH prediction
   ↓
Streamlit Web App (localhost:8501)
```

---

## Roadmap
- [x] Species classification (73 species)
- [x] Fish detection (YOLOv8)
- [x] Species identification per detection
- [x] Unknown species handling
- [x] Density estimation
- [x] Spatial heat map
- [x] Fish tracking (ByteTrack)
- [x] Water quality integration (XGBoost + FastAPI)
- [x] Integrated report
- [x] GIS mapping (Folium)
- [x] Streamlit frontend (`app.py`)
- [x] Cross-platform support (macOS MPS / CUDA / CPU)
- [ ] Mobile app
