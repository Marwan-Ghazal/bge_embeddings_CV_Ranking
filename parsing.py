#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parsing.py (semantic-friendly)
- Same parsing, but default chunking now ~280 words with 60 overlap
- Preserves section labels on chunks
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

from tqdm import tqdm

# PDF & DOCX parsing
from pypdf import PdfReader
from docx import Document

# OCR (optional fallback)
OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

SECTION_HEADINGS = [
    r"summary|objective|profile|personal\s+statement|career\s+summary|professional\s+profile",
    r"experience|work\s+experience|employment|professional\s+experience|career\s+history|technical\s+experience",
    r"projects?|project\s+experience|projects?\s*&\s*experience",
    r"skills?|technical\s+skills|key\s+skills|core\s+competencies|tools|toolbox",
    r"education|academic\s+background|qualifications",
    r"certifications?|licenses?|certificates?",
    r"publications?",
    r"awards?|honors?",
    r"languages?|language",
    r"contact|contact\s+information|details",
    r"interests|hobbies\s+and\s+interests|hobbies",
    r"extracurricular(\s+activities)?|extra\-curricular(\s+activities)?",
]
SECTION_REGEX = re.compile(r"^\s*(?P<hdr>(" + "|".join(SECTION_HEADINGS) + r"))\s*[:\-]?\s*$", re.I)

CANONICAL_MAP = {
    "work experience": "experience",
    "professional experience": "experience",
    "employment": "experience",
    "career history": "experience",
    "technical experience": "experience",
    "project experience": "projects",
    "projects & experience": "projects",
    "project & experience": "projects",
    "technical skills": "skills",
    "toolbox": "skills",
    "key skills": "skills",
    "core competencies": "skills",
    "academic background": "education",
    "qualifications": "education",
    "certificates": "certifications",
    "certificate": "certifications",
    "language": "languages",
    "personal statement": "summary",
    "professional profile": "summary",
    "career summary": "summary",
    "hobbies and interests": "interests",
    "extra-curricular activities": "extracurricular",
    "extracurricular activities": "extracurricular",
    "extracurricular": "extracurricular",
}

ALIASES = {
    "project": "projects",
    "profile": "summary",
    "details": "projects",
}

WORD_SPLIT = re.compile(r"\s+")
DATE_RNG = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\.?\s*\d{4}\s*[–\-]\s*(?:Present|Now|\d{4})",
    re.I
)

def normalize_ws(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def stitch_broken_lines(s: str) -> str:
    s = re.sub(r"(?<![.\!\?:;])\n(?=[a-z0-9])", " ", s)
    s = re.sub(r"(\w)\n(\w)", r"\1 \2", s)
    return s

def normalize_bullets(s: str) -> str:
    s = re.sub(r"[•●○·]\s*", "- ", s)
    s = re.sub(r"\n\s*-\s*", "\n- ", s)
    return s

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"^\s*[_=\-]{4,}\s*$", "", s, flags=re.M)
    s = normalize_bullets(s)
    # NOTE: keep natural hyphens intact; do NOT glue "foo - bar" -> "foo-bar"
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def drop_repeated_headers(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return text
    header_blob = "\n".join(lines[:6])[:300]
    parts = text.split("\n")
    joined = []
    seen_mid = False
    for i in range(len(parts)):
        window = "\n".join(parts[i:i+6])[:300]
        if i > 20 and not seen_mid and header_blob and window == header_blob:
            seen_mid = True
            continue
        joined.append(parts[i])
    return "\n".join(joined)

def extract_contact(full_text: str) -> dict:
    try:
        if not full_text:
            return {"emails": [], "phones": [], "links": []}
        t = full_text
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)
        raw_phones = re.findall(r"(?:(?:\+|0)\d[\d\s\-\(\)]{7,}\d)", t)
        def clean_phone(p):
            p = re.sub(r"[^\d+]", "", p)
            if re.fullmatch(r"\+?\d{9,15}", p) and not re.fullmatch(r"\d{6}", p):
                return p
        phones = sorted({p for p in (clean_phone(p) for p in raw_phones) if p})
        link_raw = re.findall(r"(https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+)", t, flags=re.I)
        def norm_link(u):
            u = u.strip().rstrip(").,;")
            return u if u.startswith("http") else f"http://{u}"
        links = sorted({norm_link(u) for u in link_raw if norm_link(u)})
        return {"emails": sorted(set(emails)), "phones": phones, "links": links}
    except Exception:
        return {"emails": [], "phones": [], "links": []}

def extract_text_from_pdf(path: str, ocr_if_needed: bool = True, max_ocr_pages: int = 10):
    text_parts, n_pages, used_ocr = [], 0, False
    try:
        reader = PdfReader(path)
        n_pages = len(reader.pages)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                text_parts.append(t)
        text = "\n".join(text_parts).strip()
    except Exception:
        text = ""
    if (not text or len(text) < 50) and ocr_if_needed and OCR_AVAILABLE:
        try:
            images = convert_from_path(path, dpi=300)
            n_pages = max(n_pages, len(images))
            text = "\n".join(pytesseract.image_to_string(img) for img in images[:max_ocr_pages])
            used_ocr = True
        except Exception:
            pass
    try:
        text = drop_repeated_headers(text or "")
        text = stitch_broken_lines(text)
        text = clean_text(text)
        text = normalize_bullets(text)
    except Exception:
        pass
    return text, n_pages, used_ocr

def extract_text_from_docx(path: str) -> str:
    try:
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        text = ""
    try:
        text = drop_repeated_headers(text or "")
        text = stitch_broken_lines(text)
        text = clean_text(text)
        text = normalize_bullets(text)
    except Exception:
        pass
    return text

def extract_text_from_file(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        txt, n_pages, used_ocr = extract_text_from_pdf(path)
        return {"text": txt, "n_pages": n_pages, "used_ocr": used_ocr}
    elif ext == ".docx":
        txt = extract_text_from_docx(path)
        return {"text": txt, "n_pages": None, "used_ocr": False}
    else:
        raise RuntimeError(f"Unsupported file type: {ext} (use PDF or DOCX)")

@dataclass
class Chunk:
    text: str
    section: str
    start_word: int
    end_word: int
    token_len_est: int

def _is_heading_token(s: str) -> bool:
    if not s:
        return False
    return bool(re.match(
        r"^\s*(summary|objective|profile|personal\s+statement|career\s+summary|professional\s+profile|"
        r"experience|work\s+experience|employment|professional\s+experience|career\s+history|technical\s+experience|"
        r"projects?|project\s+experience|projects?\s*&\s*experience|skills?|technical\s+skills|key\s+skills|"
        r"core\s+competencies|tools|toolbox|education|academic\s+background|qualifications|certifications?|"
        r"licenses?|certificates?|publications?|awards?|honors?|languages?|language|contact|contact\s+information|"
        r"details|interests|hobbies\s+and\s+interests|hobbies|extracurricular(\s+activities)?|extra\-curricular(\s+activities)?)\s*$",
        s, flags=re.I))

def split_into_sections(text: str) -> Dict[str, str]:
    if not text:
        return {}
    lines = [l.strip() for l in text.split("\n")]
    sections: Dict[str, List[str]] = {}
    current = "other"
    sections[current] = []
    for line in lines:
        m = SECTION_REGEX.match(line)
        if m:
            hdr = re.sub(r"\s+", " ", m.group("hdr").lower())
            current = CANONICAL_MAP.get(hdr, hdr)
            sections.setdefault(current, [])
            continue
        m2 = re.match(r"^\s*([A-Za-z &/]+)\s*:\s*(.+)$", line)
        if m2:
            hdr2 = re.sub(r"\s+", " ", m2.group(1).lower())
            if _is_heading_token(hdr2):
                current = CANONICAL_MAP.get(hdr2, hdr2)
                sections.setdefault(current, [])
                line = m2.group(2).strip()
        line = re.sub(r"^\s*(Outline|Summary)\b[:\-]?\s*", "", line, flags=re.I)
        if line:
            sections.setdefault(current, []).append(line)
    joined = {k: normalize_ws("\n".join(v)).strip() for k, v in sections.items()}
    joined = {k: v for k, v in joined.items() if v}
    joined = {ALIASES.get(k, k): v for k, v in joined.items()}
    return joined

def chunk_text_by_words(text: str, section: str,
                        max_words: int = 280, overlap_words: int = 60) -> List[Chunk]:
    if not text or not text.strip():
        return []
    words = [w for w in WORD_SPLIT.split(text.strip()) if w]
    if not words:
        return []
    chunks: List[Chunk] = []
    i = 0
    while i < len(words):
        j = min(i + max_words, len(words))
        sub = " ".join(words[i:j]).strip()
        if sub:
            chunks.append(Chunk(
                text=sub, section=section, start_word=i, end_word=j,
                token_len_est=int((j - i) * 0.75)
            ))
        if j == len(words):
            break
        i = max(0, j - overlap_words)
        if i >= j:
            i = j
    return chunks

def summarize_experience_dates(exp_text: str) -> dict:
    if not exp_text:
        return {"date_spans_found": [], "n_spans": 0}
    ranges = DATE_RNG.findall(exp_text or "")
    return {"date_spans_found": ranges, "n_spans": len(ranges)}

def parse_cv_file(path: str, chunk_max_words: int = 280, chunk_overlap_words: int = 60) -> Dict[str, Any]:
    meta = extract_text_from_file(path)
    raw = meta["text"]
    n_pages = meta.get("n_pages")
    used_ocr = meta.get("used_ocr", False)
    if not raw:
        raise RuntimeError(f"Failed to extract text from {path}")
    sections = split_into_sections(raw) if raw else {}
    all_chunks: List[Chunk] = []
    if sections:
        for sec_name, sec_text in sections.items():
            chunks = chunk_text_by_words(sec_text, sec_name,
                                         max_words=chunk_max_words,
                                         overlap_words=chunk_overlap_words)
            all_chunks.extend(chunks)
    else:
        chunks = chunk_text_by_words(raw, "other",
                                     max_words=chunk_max_words,
                                     overlap_words=chunk_overlap_words)
        all_chunks.extend(chunks)
    stat = os.stat(path)
    exp_meta = summarize_experience_dates(sections.get("experience", ""))
    return {
        "cv_id": os.path.basename(path),
        "source_path": path,
        "filesize_bytes": stat.st_size,
        "n_pages": n_pages,
        "full_text": raw,
        "contact": extract_contact(raw),
        "sections": sections,
        "experience_meta": exp_meta,
        "chunks": [c.__dict__ for c in all_chunks]
    }

def enumerate_cv_paths(cv_files: List[str]) -> List[str]:
    paths: List[str] = []
    for p in cv_files:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith((".pdf", ".docx")):
                        paths.append(os.path.join(root, fn))
        else:
            paths.append(p)
    uniq, seen = [], set()
    for p in paths:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def main():
    ap = argparse.ArgumentParser("Ingest & parse JDs and CVs (PDF/DOCX) into sections and ~280-word chunks.")
    jd_src = ap.add_mutually_exclusive_group(required=True)
    jd_src.add_argument("--jd-text", type=str, help="Job description text")
    jd_src.add_argument("--jd-file", type=str, help="Path to a JD .txt file")
    ap.add_argument("--cv-files", nargs="+", required=True, help="CV file(s) or directories (PDF/DOCX)")
    ap.add_argument("--out", type=str, default="parsed_output.json", help="Output JSON path")
    ap.add_argument("--chunk-max-words", type=int, default=280, help="Max words per chunk")
    ap.add_argument("--chunk-overlap-words", type=int, default=60, help="Overlap words between chunks")
    args = ap.parse_args()

    jd_text = args.jd_text.strip() if args.jd_text else open(args.jd_file, "r", encoding="utf-8").read().strip()
    cv_paths = enumerate_cv_paths(args.cv_files)
    if not cv_paths:
        print("No CVs found (PDF/DOCX).")
        return

    print(f"OCR available: {OCR_AVAILABLE}")
    print(f"Found {len(cv_paths)} CV(s). Parsing → sectioning → chunking...")

    items = []
    for path in tqdm(cv_paths):
        parsed = parse_cv_file(path, chunk_max_words=args.chunk_max_words, chunk_overlap_words=args.chunk_overlap_words)
        items.append(parsed)

    output = {"jd_text": jd_text, "cv_count": len(items), "items": items}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Wrote {args.out}")
    if not OCR_AVAILABLE:
        print("Note: OCR fallback disabled (install tesseract, pdf2image, poppler) for scanned PDFs).")

if __name__ == "__main__":
    main()
