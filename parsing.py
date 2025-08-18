#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parsing.py (enhanced)

What it does:
- Read JD (text or file)
- Read CVs (PDF/DOCX)
- Clean artifacts (decorative rules, spaced hyphens, broken lines, bullets)
- Drop duplicated header blocks in PDFs
- Extract contact info (emails/phones/links) robustly
- Split into canonical sections (expanded headings + alias mapping)
- Handle "Heading: content" single-line cases
- Compute simple experience date features
- Chunk sections into word windows (configurable)
- Include file stats: filesize, n_pages, is_ocr_used

No embeddings. No TF-IDF. No LLM calls.
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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

# -----------------------------
# Config
# -----------------------------

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

# canonical name mapping (lowercased keys)
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

# final alias fix for stray keys produced by noisy layouts
ALIASES = {
    "project": "projects",
    "profile": "summary",
    "details": "projects",  # adjust if you prefer another target
}

WORD_SPLIT = re.compile(r"\s+")

# dates inside experience (very simple)
DATE_RNG = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\.?\s*\d{4}\s*[–\-]\s*(?:Present|Now|\d{4})",
    re.I
)

# -----------------------------
# Utilities
# -----------------------------

def normalize_ws(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def stitch_broken_lines(s: str) -> str:
    # Join obvious sentence continuations created by hard line breaks
    s = re.sub(r"(?<![.\!\?:;])\n(?=[a-z0-9])", " ", s)   # lower/number continuation
    s = re.sub(r"(\w)\n(\w)", r"\1 \2", s)               # word\nword -> word word
    return s

def normalize_bullets(s: str) -> str:
    s = re.sub(r"[•●○·]\s*", "- ", s)
    # ensure dashes are treated like bullets on new lines (conservative)
    s = re.sub(r"\n\s*-\s*", "\n- ", s)
    return s

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    # remove decorative rules
    s = re.sub(r"^\s*[_=\-]{4,}\s*$", "", s, flags=re.M)
    # normalize bullets early
    s = normalize_bullets(s)
    # collapse spaced hyphens "foo - bar" -> "foo-bar"
    s = re.sub(r"\s*-\s*", "-", s)
    # bring back bullets to have a space after dash
    s = re.sub(r"\n-", "\n- ", s)
    # collapse multiple spaces and big blank blocks
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def drop_repeated_headers(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return text
    # first few lines as signature
    header_blob = "\n".join(lines[:6])[:300]
    parts = text.split("\n")
    joined = []
    seen_mid = False
    for i in range(len(parts)):
        window = "\n".join(parts[i:i+6])[:300]
        if i > 20 and not seen_mid and header_blob and window == header_blob:
            seen_mid = True
            continue  # skip duplicated header block
        joined.append(parts[i])
    return "\n".join(joined)

def extract_contact(full_text: str) -> dict:
    try:
        if not full_text:
            return {"emails": [], "phones": [], "links": []}
        
        t = full_text
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)

        # phones: match +201234567890 or 01123456789 etc.; allow (), spaces, dashes; then clean
        raw_phones = re.findall(r"(?:(?:\+|0)\d[\d\s\-\(\)]{7,}\d)", t)
        def clean_phone(p):
            try:
                p = re.sub(r"[^\d+]", "", p)  # keep + and digits
                # accept 9–15 digits; reject plain 6-digit tokens (often dates like 202507)
                if re.fullmatch(r"\+?\d{9,15}", p) and not re.fullmatch(r"\d{6}", p):
                    return p
            except Exception:
                return None
        phones = sorted({p for p in (clean_phone(p) for p in raw_phones) if p})

        # links: add http when missing; keep LinkedIn/GitHub/any http(s)
        link_raw = re.findall(r"(https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+)", t, flags=re.I)
        def norm_link(u):
            try:
                u = u.strip().rstrip(").,;")
                return u if u.startswith("http") else f"http://{u}"
            except Exception:
                return u
        links = sorted({norm_link(u) for u in link_raw if norm_link(u)})

        return {"emails": sorted(set(emails)), "phones": phones, "links": links}
    except Exception as e:
        print(f"Warning: Contact extraction failed: {e}")
        return {"emails": [], "phones": [], "links": []}

# -----------------------------
# File parsing (PDF/DOCX)
# -----------------------------

def extract_text_from_pdf(path: str, ocr_if_needed: bool = True, max_ocr_pages: int = 10) -> (str, int, bool):
    """
    Returns: (text, n_pages, used_ocr)
    """
    text_parts = []
    n_pages = 0
    used_ocr = False

    try:
        reader = PdfReader(path)
        n_pages = len(reader.pages)
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
                if t:
                    text_parts.append(t)
            except Exception as e:
                print(f"Warning: Failed to extract text from page in {path}: {e}")
                # Continue with other pages
        text = "\n".join(text_parts).strip()
    except Exception as e:
        print(f"Warning: Failed to read PDF {path}: {e}")
        text = ""

    if (not text or len(text) < 50) and ocr_if_needed and OCR_AVAILABLE:
        try:
            images = convert_from_path(path, dpi=300)
            n_pages = max(n_pages, len(images))
            ocr_texts = []
            for img in images[:max_ocr_pages]:
                try:
                    ocr_texts.append(pytesseract.image_to_string(img))
                except Exception as e:
                    print(f"Warning: OCR failed for image in {path}: {e}")
                    # Continue with other images
            text = "\n".join(ocr_texts)
            used_ocr = True
        except Exception as e:
            print(f"Warning: OCR processing failed for {path}: {e}")

    # cleanup pipeline
    try:
        text = drop_repeated_headers(text or "")
        text = stitch_broken_lines(text)
        text = clean_text(text)
        text = normalize_bullets(text)
    except Exception as e:
        print(f"Warning: Text cleanup failed for {path}: {e}")
        # Return text as-is if cleanup fails
    
    return text, n_pages, used_ocr

def extract_text_from_docx(path: str) -> str:
    try:
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"Warning: Failed to read DOCX {path}: {e}")
        text = ""
    
    # cleanup pipeline
    try:
        text = drop_repeated_headers(text or "")
        text = stitch_broken_lines(text)
        text = clean_text(text)
        text = normalize_bullets(text)
    except Exception as e:
        print(f"Warning: Text cleanup failed for {path}: {e}")
        # Return text as-is if cleanup fails
    
    return text

def extract_text_from_file(path: str) -> dict:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            txt, n_pages, used_ocr = extract_text_from_pdf(path)
            return {"text": txt, "n_pages": n_pages, "used_ocr": used_ocr}
        elif ext == ".docx":
            txt = extract_text_from_docx(path)
            return {"text": txt, "n_pages": None, "used_ocr": False}
        else:
            raise ValueError(f"Unsupported file type: {ext} (use PDF or DOCX)")
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {path}: {e}")

# -----------------------------
# Sectioning & Chunking
# -----------------------------

@dataclass
class Chunk:
    text: str
    section: str
    start_word: int
    end_word: int
    token_len_est: int  # rough estimate (~0.75 * words)

def _is_heading_token(s: str) -> bool:
    try:
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
    except Exception as e:
        print(f"Warning: Heading token check failed: {e}")
        return False

def split_into_sections(text: str) -> Dict[str, str]:
    """
    Heuristic: treat lines that match SECTION_REGEX as headings.
    Also handle "Heading: content" lines and drop filler tokens like 'Outline'/'Summary' prefixes.
    """
    try:
        if not text:
            return {}
        
        lines = [l.strip() for l in text.split("\n")]
        sections: Dict[str, List[str]] = {}
        current = "other"
        sections[current] = []

        for line in lines:
            try:
                # Heading-only line?
                m = SECTION_REGEX.match(line)
                if m:
                    hdr = re.sub(r"\s+", " ", m.group("hdr").lower())
                    current = CANONICAL_MAP.get(hdr, hdr)
                    sections.setdefault(current, [])
                    continue

                # "Heading: content" on one line
                m2 = re.match(r"^\s*([A-Za-z &/]+)\s*:\s*(.+)$", line)
                if m2:
                    hdr2 = re.sub(r"\s+", " ", m2.group(1).lower())
                    if _is_heading_token(hdr2):
                        current = CANONICAL_MAP.get(hdr2, hdr2)
                        sections.setdefault(current, [])
                        line = m2.group(2).strip()

                # Remove filler tokens like "Outline" or "Summary" prefixes
                line = re.sub(r"^\s*(Outline|Summary)\b[:\-]?\s*", "", line, flags=re.I)

                # Append non-empty lines
                if line:
                    sections.setdefault(current, []).append(line)
            except Exception as e:
                print(f"Warning: Failed to process line in sectioning: {e}")
                # Continue with next line

        # Join, normalize whitespace, and drop empties
        try:
            joined = {k: normalize_ws("\n".join(v)).strip() for k, v in sections.items()}
            joined = {k: v for k, v in joined.items() if v}

            # alias fix for stray keys
            joined = {ALIASES.get(k, k): v for k, v in joined.items()}
            return joined
        except Exception as e:
            print(f"Warning: Failed to join sections: {e}")
            return {}
    except Exception as e:
        print(f"Warning: Section splitting failed: {e}")
        return {}

def chunk_text_by_words(text: str, section: str,
                        max_words: int = 900, overlap_words: int = 200) -> List[Chunk]:
    try:
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
                    text=sub,
                    section=section,
                    start_word=i,
                    end_word=j,
                    token_len_est=int((j - i) * 0.75)
                ))
            if j == len(words):
                break
            i = max(0, j - overlap_words)
            if i >= j:
                i = j
        return chunks
    except Exception as e:
        print(f"Warning: Failed to chunk text for section {section}: {e}")
        return []

def summarize_experience_dates(exp_text: str) -> dict:
    try:
        if not exp_text:
            return {"date_spans_found": [], "n_spans": 0}
        ranges = DATE_RNG.findall(exp_text or "")
        return {"date_spans_found": ranges, "n_spans": len(ranges)}
    except Exception as e:
        print(f"Warning: Experience date extraction failed: {e}")
        return {"date_spans_found": [], "n_spans": 0}

def parse_cv_file(path: str,
                  chunk_max_words: int = 900,
                  chunk_overlap_words: int = 200) -> Dict[str, Any]:
    try:
        meta = extract_text_from_file(path)
        raw = meta["text"]
        n_pages = meta.get("n_pages")
        used_ocr = meta.get("used_ocr", False)

        if not raw:
            raise ValueError(f"Failed to extract text from {path}")

        sections = split_into_sections(raw) if raw else {}
        all_chunks: List[Chunk] = []

        if sections:
            for sec_name, sec_text in sections.items():
                try:
                    chunks = chunk_text_by_words(
                        sec_text, sec_name,
                        max_words=chunk_max_words,
                        overlap_words=chunk_overlap_words
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Warning: Failed to chunk section {sec_name} from {path}: {e}")
                    # Continue with other sections
        elif raw:
            try:
                chunks = chunk_text_by_words(
                    raw, "other",
                    max_words=chunk_max_words,
                    overlap_words=chunk_overlap_words
                )
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Warning: Failed to chunk text from {path}: {e}")
                # Continue without chunks

        stat = os.stat(path)
        exp_meta = summarize_experience_dates(sections.get("experience", ""))

        return {
            "cv_id": os.path.basename(path),
            "source_path": path,
            "filesize_bytes": stat.st_size,
            "n_pages": n_pages,
            "full_text": raw,
            "contact": extract_contact(raw),
            "sections": sections,                      # {section_name: section_text}
            "experience_meta": exp_meta,               # simple date signals
            "chunks": [c.__dict__ for c in all_chunks] # [{text, section, start_word, end_word, token_len_est}]
        }
    except Exception as e:
        raise RuntimeError(f"Failed to parse CV file {path}: {e}")

# -----------------------------
# CLI & I/O
# -----------------------------

def enumerate_cv_paths(cv_files: List[str]) -> List[str]:
    paths: List[str] = []
    try:
        for p in cv_files:
            try:
                if os.path.isdir(p):
                    for root, _, files in os.walk(p):
                        for fn in files:
                            if fn.lower().endswith((".pdf", ".docx")):
                                paths.append(os.path.join(root, fn))
                else:
                    paths.append(p)
            except Exception as e:
                print(f"Warning: Failed to process path {p}: {e}")
                # Continue with other paths
        
        # de-dup (preserve order)
        uniq, seen = [], set()
        for p in paths:
            if p not in seen:
                uniq.append(p); seen.add(p)
        return uniq
    except Exception as e:
        print(f"Warning: CV path enumeration failed: {e}")
        return []

def main():
    ap = argparse.ArgumentParser("Ingest & parse JDs and CVs (PDF/DOCX) into canonical sections and chunks. No embeddings/LLM.")
    jd_src = ap.add_mutually_exclusive_group(required=True)
    jd_src.add_argument("--jd-text", type=str, help="Job description text")
    jd_src.add_argument("--jd-file", type=str, help="Path to a JD .txt file")
    ap.add_argument("--cv-files", nargs="+", required=True, help="CV file(s) or directories (PDF/DOCX)")
    ap.add_argument("--out", type=str, default="parsed_output.json", help="Output JSON path")
    ap.add_argument("--chunk-max-words", type=int, default=900, help="Max words per chunk")
    ap.add_argument("--chunk-overlap-words", type=int, default=200, help="Overlap words between chunks")
    args = ap.parse_args()

    try:
        jd_text = args.jd_text.strip() if args.jd_text else open(args.jd_file, "r", encoding="utf-8").read().strip()
    except Exception as e:
        print(f"Error: Failed to read JD text: {e}")
        return
    try:
        cv_paths = enumerate_cv_paths(args.cv_files)
        if not cv_paths:
            print("No CVs found (PDF/DOCX).")
            return
    except Exception as e:
        print(f"Error: Failed to enumerate CV paths: {e}")
        return

    print(f"OCR available: {OCR_AVAILABLE}")
    print(f"Found {len(cv_paths)} CV(s). Parsing → sectioning → chunking...")

    items = []
    for path in tqdm(cv_paths):
        try:
            parsed = parse_cv_file(
                path,
                chunk_max_words=args.chunk_max_words,
                chunk_overlap_words=args.chunk_overlap_words
            )
            items.append(parsed)
        except Exception as e:
            print(f"Warning: Failed to parse {path}: {e}")
            # Continue with other files

    output = {
        "jd_text": jd_text,
        "cv_count": len(items),
        "items": items
    }

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error: Failed to write output file {args.out}: {e}")
        return

    print(f"\nDone. Wrote {args.out}")
    if not OCR_AVAILABLE:
        print("Note: OCR fallback disabled (install tesseract, pdf2image, poppler) for scanned PDFs.")

if __name__ == "__main__":
    main()
