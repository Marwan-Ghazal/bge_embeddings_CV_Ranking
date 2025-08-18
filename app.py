# app.py — FastAPI for CV ranking with BGE (Clean Version)
import os, shutil, tempfile
from datetime import datetime
from typing import List, Optional
import socket

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import parsing as P  # your parsing.py
from improved_ranker import improved_rank_cv, get_model  # improved ranking system
from pathlib import Path

# ------------------------
# Get local IP address
# ------------------------
def get_local_ip():
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

# ------------------------
# Config
# ------------------------
# Recommended smaller chunks for BGE (less truncation)
CHUNK_MAX = 900
CHUNK_OVL = 200

DEFAULT_MODEL = "bge-base-en-v1.5"  # Use local model folder

app = FastAPI(title="CV Ranker API", version="2.0")

# CORS (loose for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Static files (optional UI)
# ------------------------
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR  = BASE_DIR / "web"

@app.get("/", include_in_schema=False)
def serve_index():
    path = WEB_DIR / "index.html"
    if not path.exists():
        raise HTTPException(404, f"index.html not found at {path}")
    return FileResponse(path)

@app.get("/results.js", include_in_schema=False)
def serve_results_js():
    path = WEB_DIR / "results.js"
    if not path.exists():
        raise HTTPException(404, f"results.js not found at {path}")
    return FileResponse(path)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# ------------------------
# Health & models
# ------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now().isoformat(timespec="seconds")}

@app.get("/models")
def models():
    return {
        "default_model": DEFAULT_MODEL,
        "ranking_method": "improved_bi_encoder",
        "description": "Improved BGE bi-encoder ranking with better parsing and scoring"
    }

@app.get("/ip")
def get_ip():
    """Return the machine's local LAN IP to help access the server from other devices."""
    return {"ip": get_local_ip()}

# ------------------------
# Clean ranking system - no more corrupted normalization
# ------------------------

# ------------------------
# Main ranking endpoint
# ------------------------
@app.post("/rank")
async def rank(
    email: str = Form(...),
    jd_text: str = Form(...),
    cv_file: UploadFile = File(...),
):
    """
    Improved CV ranking using enhanced BGE bi-encoder system.
    Better parsing, chunking, and semantic scoring for accurate results.
    """
    if not email:
        raise HTTPException(400, "email is required")
    if not jd_text:
        raise HTTPException(400, "jd_text is required")
    if not cv_file:
        raise HTTPException(400, "cv_file is required (.pdf/.docx)")

    tmpdir = tempfile.mkdtemp(prefix="cvrank_")
    try:
        # Save CV file
        cv_name = cv_file.filename or "cv"
        if not cv_name.lower().endswith((".pdf", ".docx")):
            raise HTTPException(400, "CV file must be .pdf or .docx")
        
        cv_path = os.path.join(tmpdir, cv_name)
        with open(cv_path, "wb") as out:
            shutil.copyfileobj(cv_file.file, out)

        # Rank via improved ranking system (handles parsing internally)
        try:
            score, cv_data, explanation = improved_rank_cv(
                jd_text, cv_path, model_name=DEFAULT_MODEL
            )
        except Exception as e:
            raise HTTPException(500, f"Ranking failed: {str(e)}")

        # Normalize types for JSON
        try:
            score_val = float(score)
        except Exception:
            score_val = float(score.item()) if hasattr(score, "item") else 0.0

        rows = [{
            "cv_id": cv_data.get("cv_id", "unknown"),
            "score": score_val,
            "source_path": cv_data.get("source_path", ""),
            "email": email,
            "phone": ",".join(cv_data.get("contact", {}).get("phones", [])),
            "links": ",".join(cv_data.get("contact", {}).get("links", [])),
            "chunk_count": len(cv_data.get("chunks", [])),
            "sections_found": list(cv_data.get("sections", {}).keys()),
            "explanation": explanation
        }]

        return {
            "model": DEFAULT_MODEL,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "email": email,
            "jd_text": jd_text,
            "rows": rows,
            "warning": None,
            "cv_count": 1,
            "ranking_method": "improved_bi_encoder",
            "description": "Enhanced BGE ranking with improved parsing and semantic scoring"
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ------------------------
# Server startup
# ------------------------
if __name__ == "__main__":
    import uvicorn
    
    # Get local IP address
    local_ip = get_local_ip()
    port = 8000
    
    print("🚀 Starting CV Ranker Server (Clean Version)...")
    print(f"📍 Local URL: http://127.0.0.1:{port}")
    print(f"🌐 Network URL: http://{local_ip}:{port}")
    print("📖 API Docs: http://127.0.0.1:8000/docs")
    print("🔧 Press Ctrl+C to stop the server")
    print("✅ Using clean, honest ranking system")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "app:app",
            host=local_ip,  # Bind directly to your machine's LAN IP
            port=port,
            reload=False,  # Set to True for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
