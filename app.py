import streamlit as st
import cv2
import numpy as np
import os
import sys
import json
import torch
import tempfile
from PIL import Image
from collections import Counter
from ultralytics import YOLO

# Set page config
st.set_page_config(page_title="ML Fish Classifier & Ecosystem AI", page_icon="🐟", layout="wide")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo_fish", "weights", "best.pt")
EFFNET_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
CLASSES_PATH = os.path.join(PROJECT_ROOT, "models", "class_names.json")
CLASSES_31_PATH = os.path.join(PROJECT_ROOT, "models", "class_names_31.json")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@st.cache_resource
def load_yolo():
    return YOLO(YOLO_MODEL_PATH)

@st.cache_resource
def load_classifier():
    import sys
    sys.path.append(os.path.join(PROJECT_ROOT, "src", "classification"))
    from model import build_model
    
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    
    state_dict = torch.load(EFFNET_MODEL_PATH, map_location=DEVICE)
    ckpt_classes = state_dict["classifier.1.weight"].shape[0]
    if ckpt_classes == 31 and os.path.exists(CLASSES_31_PATH):
        with open(CLASSES_31_PATH) as f:
            class_names = json.load(f)
            
    model = build_model(num_classes=ckpt_classes)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model, class_names

st.title("🐟 ML Fish Classifier & Eco-Dashboard")
st.markdown("AI-based fish species identification, detection, counting, density estimation & water quality analysis.")

sidebar_mode = st.sidebar.selectbox("Choose Mode", [
    "Fish Detection & Counting (YOLOv8)",
    "Species Classification (EfficientNetB0)",
    "Density Estimation",
    "Spatial Heat Map",
    "Water Quality Analysis"
])

st.sidebar.markdown("### 🎚️ Detection Settings")
conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10, max_value=0.95, value=0.25, step=0.05,
    help="Lower = detects more fish (good for schools/groups). Raise if you get false positives."
)
iou_thresh = st.sidebar.slider(
    "IoU / NMS Threshold",
    min_value=0.10, max_value=0.95, value=0.30, step=0.05,
    help="For schools of fish, keep this LOW (0.20–0.35) so nearby fish are not merged into one box."
)
max_det = st.sidebar.slider(
    "Max Detections",
    min_value=10, max_value=500, value=300, step=10,
    help="Maximum number of fish the model can detect in one image. Raise for very large schools."
)
st.sidebar.caption(f"🔍 Conf: `{conf_thresh:.2f}` | NMS IoU: `{iou_thresh:.2f}` | Max: `{max_det}`")

# Image selection
st.sidebar.markdown("---")
use_sample = st.sidebar.checkbox("Use sample fish image", value=True)
uploaded_file = st.sidebar.file_uploader("Or Upload Fish Image", type=["jpg", "jpeg", "png"])

img_path = None
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp")
    img.save(tmp.name)
    img_path = tmp.name
elif use_sample and os.path.exists(os.path.join(PROJECT_ROOT, "sample_fish.jpg")):
    img_path = os.path.join(PROJECT_ROOT, "sample_fish.jpg")

if img_path is None:
    st.info("Please upload an image or check 'Use sample fish image'.")
    st.stop()

# ----------------- Mode 1: Fish Detection & Counting -----------------
if sidebar_mode == "Fish Detection & Counting (YOLOv8)":
    st.subheader("Fish Detection & Counting (YOLOv8)")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_path, caption="Original Image", width="stretch")
    
    with st.spinner("Detecting fish..."):
        yolo = load_yolo()
        results = yolo.predict(source=img_path, conf=conf_thresh, iou=iou_thresh, max_det=max_det, agnostic_nms=True, verbose=False)
        res = results[0]
        res_bgr = res.plot()  # plots boxes and labels
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        fish_count = len(res.boxes)
        
    with col2:
        st.image(res_rgb, caption=f"Detection Result ({fish_count} fish detected)", width="stretch")

    if fish_count == 1:
        st.success(f"✅ **1 fish detected**")
    elif fish_count == 0:
        st.warning("⚠️ No fish detected. Try lowering the **Confidence Threshold** in the sidebar.")
    else:
        st.success(f"✅ **{fish_count} fish detected**")

    if fish_count > 0:
        confs = [float(b.conf[0]) * 100 for b in res.boxes]
        st.write(f"Confidence — Avg: `{np.mean(confs):.1f}%` | Min: `{np.min(confs):.1f}%` | Max: `{np.max(confs):.1f}%`")

    st.info("💡 **Missing fish in a school?** Lower **Confidence Threshold** (try 0.15–0.25) and lower **IoU Threshold** (try 0.20–0.30). "
            "**Getting duplicate boxes?** Raise the **Confidence Threshold** (try 0.40–0.60).")

# ----------------- Mode 2: Species Classification -----------------
elif sidebar_mode == "Species Classification (EfficientNetB0)":
    st.subheader("Species Classification (EfficientNetB0)")
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_path, caption="Input Fish", width="stretch")
        
    with st.spinner("Identifying species..."):
        model, class_names = load_classifier()
        pil_img = Image.open(img_path).convert("RGB")
        tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top5 = torch.topk(probs, k=5)
            
    with col2:
        st.markdown(f"### Top Prediction: **{class_names[top5.indices[0]]}** ({top5.values[0]*100:.2f}%)")
        st.markdown("#### Top 5 Candidates:")
        table_data = []
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            table_data.append({"Rank": i+1, "Species": class_names[idx], "Confidence (%)": f"{prob*100:.2f}%"})
        st.table(table_data)

# ----------------- Mode 3: Density Estimation -----------------
elif sidebar_mode == "Density Estimation":
    st.subheader("Fish Density Estimation")
    colA, colB, colC = st.columns(3)
    with colA:
        depth_m = st.number_input("Camera Depth (meters)", 0.5, 20.0, 2.0, 0.5)
    with colB:
        fov_deg = st.number_input("Camera FOV (degrees)", 30.0, 160.0, 90.0, 5.0)
    with colC:
        pond_area = st.number_input("Total Pond Area (m²)", 100.0, 100000.0, 2000.0, 100.0)
        
    import math
    fov_rad = math.radians(fov_deg)
    width = 2 * depth_m * math.tan(fov_rad / 2)
    height = width / (4 / 3)
    cam_area = width * height
    
    yolo = load_yolo()
    res = yolo.predict(source=img_path, conf=conf_thresh, iou=iou_thresh, max_det=max_det, agnostic_nms=True, verbose=False)[0]
    total_fish = len(res.boxes)
    density = total_fish / cam_area
    est_pond_pop = int(density * pond_area)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Camera Coverage Area", f"{cam_area:.2f} m²")
    m2.metric("Detected Fish", f"{total_fish}")
    m3.metric("Areal Density", f"{density:.3f} fish/m²")
    m4.metric("Est. Total Pond Fish", f"{est_pond_pop:,}")
    
    st.image(res.plot()[:, :, ::-1], caption="Detected Fish for Density Sample", width="stretch")

# ----------------- Mode 4: Spatial Heat Map -----------------
elif sidebar_mode == "Spatial Heat Map":
    st.subheader("Fish Spatial Density Heat Map")
    yolo = load_yolo()
    res = yolo.predict(source=img_path, conf=conf_thresh, iou=iou_thresh, max_det=max_det, agnostic_nms=True, verbose=False)[0]
    image_bgr = cv2.imread(img_path)
    h, w = image_bgr.shape[:2]
    density_map = np.zeros((h, w), dtype=np.float32)
    
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(density_map, (cx, cy), radius=int(min(w, h) * 0.05), color=1.0, thickness=-1)
        
    blurred = cv2.GaussianBlur(density_map, (0, 0), sigmaX=min(w, h) * 0.04)
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    heatmap_gray = np.uint8(255 * blurred)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.65, heatmap_color, 0.35, 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Density Heatmap Overlay", width="stretch")
    with col2:
        st.image(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), caption="Heatmap Only", width="stretch")

# ----------------- Mode 5: Water Quality -----------------
elif sidebar_mode == "Water Quality Analysis":
    st.subheader("Water Potability & Quality Assessment")
    col1, col2, col3 = st.columns(3)
    with col1:
        turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 3.9, 0.1)
    with col2:
        hardness = st.slider("Hardness (mg/L)", 50.0, 400.0, 196.0, 5.0)
    with col3:
        tds = st.slider("TDS (ppm)", 100.0, 2000.0, 500.0, 20.0)
        
    from water_quality.main import predict as predict_wq, WaterInput
    res = predict_wq(WaterInput(Turbidity=turbidity, Hardness=hardness, TDS=tds))
    
    r1, r2 = st.columns(2)
    with r1:
        if res["safe"]:
            st.success(f"### Result: {res['message']}")
        else:
            st.error(f"### Result: {res['message']}")
    with r2:
        st.info(f"### Predicted pH: **{res['predicted_ph']}**")
