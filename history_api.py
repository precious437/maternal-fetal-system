import os
import psycopg2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

# --------------------
# Database Setup
# --------------------
DATABASE_URL = os.environ.get("SUPABASE_DB_URL")  # Supabase PostgreSQL
conn = psycopg2.connect(DATABASE_URL)

MONGO_URI = os.environ.get("MONGO_URI")           # MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client["maternal_ai"]
detections_collection = db["detections"]

# --------------------
# FastAPI App
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------------
# History Endpoint
# --------------------
@app.get("/history")
def get_history():
    """
    Return all scans with YOLO detections, educational tips, and Oracle-rule alerts
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT id, filename, url, uploaded_by, created_at FROM scans ORDER BY created_at DESC"
    )
    scans = cur.fetchall()
    cur.close()

    history = []

    for scan in scans:
        scan_id, filename, url, user_id, created_at = scan

        # Get detections from MongoDB
        detection_doc = detections_collection.find_one({"scan_id": scan_id})
        if detection_doc:
            detections = detection_doc.get("detections", [])
            alerts = detection_doc.get("alerts", [])
        else:
            detections = []
            alerts = []

        history.append({
            "scan_id": scan_id,
            "filename": filename,
            "url": url,
            "user_id": user_id,
            "created_at": str(created_at),
            "detections": detections,
            "alerts": alerts
        })

    return history