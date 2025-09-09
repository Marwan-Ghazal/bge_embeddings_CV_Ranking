# 🎯 BGE CV Ranking System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![BGE](https://img.shields.io/badge/BGE-Embeddings-orange.svg)](https://huggingface.co/BAAI/bge-base-en-v1.5)

A **local-first CV ranking service** that uses state-of-the-art BGE (BAAI General Embedding) models to semantically match job descriptions with candidate CVs. Features a modern web interface, REST API, and intelligent scoring with calibration.

## ✨ Features

- 🚀 **Fast Semantic Matching** - BGE embeddings for accurate CV-JD similarity scoring
- 🌐 **Modern Web Interface** - Clean, responsive UI with real-time ranking visualization
- 🔧 **REST API** - Full FastAPI integration with automatic documentation
- 📄 **Multi-format Support** - PDF and DOCX CV parsing with fallback mechanisms
- 🎯 **Smart Chunking** - Optimized text segmentation (280 words, 60 overlap) for better embeddings
- 📊 **Score Calibration** - Monotonic calibration for more interpretable similarity scores
- 🏠 **Local-first** - Prioritizes local models, falls back to Hugging Face when needed
- 🔍 **Section-aware Parsing** - Intelligent CV section detection and scoring boosts
- 📈 **Interactive Charts** - Visual ranking results with Chart.js integration

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI Server │    │  BGE Embeddings │
│   (index.html)  │◄──►│     (app.py)     │◄──►│   Model Layer   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────▼────────┐
                       │  Core Components │
                       │                 │
                       │ • parsing.py    │
                       │ • improved_     │
                       │   ranker.py     │
                       │ • calibration.py│
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bge_embeddings_CV_Ranking
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # Linux/macOS
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r req.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

The server will start and display:
```
🚀 Starting CV Ranker Server (Clean Version)...
📍 Local URL: http://127.0.0.1:8000
🌐 Network URL: http://192.168.1.100:8000
📖 API Docs: http://127.0.0.1:8000/docs
```

## 💻 Usage

### Web Interface

1. Navigate to `http://127.0.0.1:8000`
2. Enter your email address
3. Paste the job description text
4. Upload a CV file (.pdf or .docx)
5. Click "Rank CV" to get semantic similarity scores

### API Usage

#### Rank a Single CV

**Endpoint:** `POST /rank`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `email` (string, required): Contact email
- `jd_text` (string, required): Job description text
- `cv_file` (file, required): CV file (.pdf or .docx)

**Example using curl:**

```bash
# Windows PowerShell
curl.exe -X POST "http://127.0.0.1:8000/rank" `
  -F "email=recruiter@company.com" `
  -F "jd_text=Senior Python Developer with 5+ years experience in ML and FastAPI" `
  -F "cv_file=@path/to/resume.pdf;type=application/pdf"

# Linux/macOS
curl -X POST "http://127.0.0.1:8000/rank" \
  -F "email=recruiter@company.com" \
  -F "jd_text=Senior Python Developer with 5+ years experience in ML and FastAPI" \
  -F "cv_file=@path/to/resume.pdf;type=application/pdf"
```

**Response:**
```json
{
  "model": "bge-base-en-v1.5",
  "generated_at": "2024-01-15T10:30:00",
  "email": "recruiter@company.com",
  "jd_text": "Senior Python Developer...",
  "rows": [
    {
      "cv_id": "resume.pdf",
      "score": 0.847,
      "source_path": "/tmp/resume.pdf",
      "email": "candidate@email.com",
      "phone": "+1-555-0123",
      "links": "linkedin.com/in/candidate",
      "chunk_count": 12,
      "sections_found": ["experience", "skills", "education"],
      "explanation": "Strong semantic match | calibration=piecewise, raw=0.723, cal=0.847"
    }
  ],
  "score_raw": 0.723,
  "score_0_100": 84.7,
  "calibration": "piecewise",
  "cv_count": 1,
  "ranking_method": "semantic (calibrated)"
}
```

### Other Endpoints

- **Health Check:** `GET /health`
- **Model Info:** `GET /models`
- **Server IP:** `GET /ip`
- **API Documentation:** `GET /docs` (Swagger UI)

## 🔧 Configuration

### Model Configuration

The system uses a **local-first approach** for BGE models:

1. **Local Models** (recommended for speed):
   ```
   project_root/
   ├── bge-base-en-v1.5/     # Local model folder
   │   ├── config.json
   │   ├── pytorch_model.bin
   │   └── ...
   ```

2. **Remote Models** (automatic fallback):
   - `BAAI/bge-base-en-v1.5` (default)
   - `BAAI/bge-large-en-v1.5`
   - `BAAI/bge-m3`

### Chunking Parameters

Optimized for BGE embeddings:
- **Chunk Size:** 280 words
- **Overlap:** 60 words
- **Section-aware:** Preserves CV section context

## 📊 Scoring System

### Semantic Similarity

1. **Text Preprocessing:** Clean and chunk CV/JD text
2. **Embedding Generation:** BGE model encodes text chunks
3. **Similarity Calculation:** Cosine similarity between embeddings
4. **Section Boosting:** Enhanced scoring for relevant CV sections
5. **Score Calibration:** Monotonic mapping for better interpretability

### Calibration Curve

Raw similarity scores are calibrated using a piecewise linear function:

```
Raw Score → Calibrated Score
0.30      → 0.40  (Poor match)
0.45      → 0.68  (Borderline)
0.50      → 0.80  (Good match)
0.60      → 0.93  (Excellent match)
```

## 📁 Project Structure

```
bge_embeddings_CV_Ranking/
├── web/
│   └── index.html              # Modern web interface
├── app.py                      # FastAPI server & endpoints
├── improved_ranker.py          # Core ranking algorithm
├── parsing.py                  # CV/PDF text extraction
├── calibration.py              # Score calibration system
├── downloadmodel.py            # Model download utilities
├── req.txt                     # Python dependencies
└── README.md                   # This file
```

## 🛠️ Development

### Adding New Models

1. Update `REMOTE_MODEL_MAP` in `improved_ranker.py`:
   ```python
   REMOTE_MODEL_MAP = {
       "your-model": "huggingface/model-repo",
       # ...
   }
   ```

2. Modify `DEFAULT_MODEL` in `app.py` if needed

### Customizing Scoring

Edit `improved_ranker.py` to adjust:
- Section importance weights
- Chunking strategies  
- Similarity aggregation methods

### UI Customization

The web interface (`web/index.html`) uses:
- Modern CSS with dark theme
- Chart.js for visualizations
- Responsive design principles

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Fails**
   ```bash
   # Pre-download models
   python downloadmodel.py
   ```

2. **PDF Parsing Issues**
   ```bash
   # Install additional dependencies
   pip install pytesseract pdf2image
   ```

3. **Memory Issues with Large CVs**
   - Reduce `CHUNK_MAX` in `app.py`
   - Use smaller BGE model variant

### Debug Mode

Enable detailed logging:
```bash
python app.py --log-level debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BAAI](https://github.com/FlagOpen/FlagEmbedding) for BGE embeddings
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Sentence Transformers](https://www.sbert.net/) for embedding utilities

## 📞 Support

For questions, issues, or contributions:
- Open an [Issue](../../issues)
- Check the [API Documentation](http://127.0.0.1:8000/docs) when running locally
- Review the code comments for implementation details

---

**Built with ❤️ for efficient, local-first CV ranking**
