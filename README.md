How to start

.venv\Scripts\activate

pip install -r req.txt

python app.py



Server URLs
- Root UI: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs
- Health:   GET /health
- Models:   GET /models
- Your LAN IP: GET /ip (server binds to your LAN IP so other devices can access it)


Main endpoint (rank a single CV)
- POST /rank (multipart/form-data)
  - Fields:
    - email: string
    - jd_text: string (job description)
    - cv_file: file (.pdf or .docx)

Example (PowerShell curl)
curl.exe -s -D - \
  -F "email=test@example.com" \
  -F "jd_text=AI Developer role requiring Python and ML" \
  -F "cv_file=@data/cvs/Resume-MarwanGhazal.pdf;type=application/pdf" \
  http://127.0.0.1:8000/rank


Model resolution (local-first, remote fallback)
- The server tries to load a local folder model first (e.g., bge-base-en-v1.5/)
- If not found locally, it falls back to the Hugging Face repo (e.g., BAAI/bge-base-en-v1.5)


Notes
- Input parsing and chunking handled by parsing.py (with fallbacks)
- Ranking uses improved bi-encoder scoring in improved_ranker.py
