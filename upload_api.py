import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend origin
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    # Upload file to Supabase Storage
    res = supabase.storage.from_("scans").upload(file.filename, contents)
    url = supabase.storage.from_("scans").get_public_url(file.filename).public_url
    return {"url": url, "filename": file.filename}