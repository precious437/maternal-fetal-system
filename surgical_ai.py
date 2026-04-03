import os
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from ultralytics import YOLO

# --------------------
# MongoDB Setup
# --------------------
MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["maternal_ai"]
detections_collection = db["detections"]

# --------------------
# YOLO Model
# --------------------
model = YOLO("yolov8n.pt")  # Replace with custom model later

# --------------------
# FastAPI App
# --------------------
app = FastAPI()

# --------------------
# Pydantic model for request
# --------------------
class DetectRequest(BaseModel):
    filename: str
    url: str
    scan_id: int

# --------------------
# Educational tips mapping
# --------------------
EDU_MAPPING = {
    0: "Fetal head: review anatomy and measurement techniques.",
    1: "Fetal heart: check chambers, valves, and blood flow.",
    2: "Placenta: check location, thickness, and grading.",
    3: "Umbilical cord: inspect insertion and vessels.",
    4: "Fetal limbs: confirm movement and development.",
    5: "Amniotic fluid: assess volume and clarity."
}

# --------------------
# Oracle rules
# --------------------
def apply_oracle_rules(detections):
    """
    Apply simple rule-based checks on the detections.
    Returns a list of alerts or recommendations.
    """
    alerts = []

    for det in detections:
        cls = det["class"]
        conf = det["confidence"]
        bbox = det["bbox"]
        # Example rule: high confidence on fetal head → check head size
        if cls == 0 and conf > 0.9:
            width = bbox[0][2] - bbox[0][0]
            if width > 120:
                alerts.append("Alert: Fetal head appears large — review for macrocephaly.")
            elif width < 60:
                alerts.append("Alert: Fetal head appears small — review for microcephaly.")

        # Example rule: amniotic fluid detection
        if cls == 5:
            height = bbox[0][3] - bbox[0][1]
            if height < 50:
                alerts.append("Alert: Low amniotic fluid — check for oligohydramnios.")
            elif height > 200:
                alerts.append("Alert: High amniotic fluid — check for polyhydramnios.")

    return alerts

# --------------------
# Detection Endpoint
# --------------------
@app.post("/detect")
def detect(req: DetectRequest):
    """
    Run YOLO detection, add educational tips, and apply Oracle-style rules
    """
    # Run YOLO
    results = model(req.url)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            det = {
                "class": cls,
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy.tolist(),
                "education": EDU_MAPPING.get(cls, "No explanation available")
            }
            detections.append(det)

    # Apply Oracle-style rules
    alerts = apply_oracle_rules(detections)

    # Save to MongoDB
    detections_collection.insert_one({
        "scan_id": req.scan_id,
        "filename": req.filename,
        "detections": detections,
        "alerts": alerts
    })

    # Return everything to frontend
    return {
        "scan_id": req.scan_id,
        "filename": req.filename,
        "detections": detections,
        "alerts": alerts
    }