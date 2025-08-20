#!/usr/bin/env python3
"""
Semantic-focused CV ranking
- BGE prompt fix
- 280/60 chunking via parsing.py
- JD -> 6–12 semantic queries (whole + light sections + top NPs)
- Section priors & CV section boosts
- Top-k pooling per chunk; weighted-avg + best
- Batched, normalized embeddings
"""

import os
import re
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import parsing as P
from datetime import datetime

_MODELS: Dict[str, SentenceTransformer] = {}

REMOTE_MODEL_MAP = {
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
}

def get_model(name: str = "bge-base-en-v1.5") -> SentenceTransformer:
    if name in _MODELS:
        return _MODELS[name]
    for path in [name, name.rstrip("/") + "/", "bge-base-en-v1.5", "bge-base-en-v1.5/"]:
        if os.path.isdir(path):
            _MODELS[name] = SentenceTransformer(path)
            print(f"✅ Loaded local model: {path}")
            return _MODELS[name]
    repo_id = REMOTE_MODEL_MAP.get(name, name)
    _MODELS[name] = SentenceTransformer(repo_id)
    print(f"🌐 Loaded remote model from HF: {repo_id}")
    return _MODELS[name]

# -----------------------------
# JD query construction+    
# -----------------------------

HEAD_CANON = {
    "requirements": "requirements",
    "must have": "requirements",
    "must-have": "requirements",
    "qualifications": "requirements",
    "skills": "skills",
    "responsibilities": "responsibilities",
    "duties": "responsibilities",
    "what you will do": "responsibilities",
    "about": "about",
    "about us": "about",
    "summary": "summary",
    "objective": "summary",
    "profile": "summary",
    "preferred": "preferred",
    "nice to have": "preferred",
}

HEAD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z \-/']{1,40})\s*:?\s*$")

STOP = set("""
a an and are as at be by for from has have in into is it its of on or our that the their this to with you your we they will
""".split())

WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9+./#\-]*")

def _light_split_sections(text: str) -> Dict[str, str]:
    lines = [l.rstrip() for l in text.splitlines()]
    sections: Dict[str, List[str]] = {}
    current = "other"
    sections[current] = []
    for ln in lines:
        m = HEAD_RE.match(ln.strip().lower())
        if m:
            raw = m.group(1).strip().lower()
            can = HEAD_CANON.get(raw, raw)
            current = can
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(ln)
    return {k: "\n".join(v).strip() for k, v in sections.items() if v and "".join(v).strip()}

def _top_np_phrases(text: str, k: int = 8) -> List[str]:
    # Simple noun-phrase-ish mining: keep 2-5 token spans, drop stopwords-only spans.
    tokens = [t.lower() for t in WORD.findall(text)]
    # build candidate ngrams
    cands = {}
    for n in (2, 3, 4, 5):
        for i in range(len(tokens) - n + 1):
            span = tokens[i:i+n]
            if all(t in STOP for t in span): 
                continue
            # keep spans with at least one non-stop token
            if any(t not in STOP for t in span):
                phrase = " ".join(span)
                cands[phrase] = cands.get(phrase, 0) + 1
    # sort by frequency then length (desc)
    ranked = sorted(cands.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    out = []
    seen = set()
    for p, _ in ranked:
        if p in seen: 
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= k:
            break
    return out

def build_jd_queries(jd_text: str, min_q: int = 6, max_q: int = 12) -> Tuple[List[str], np.ndarray]:
    """Return (queries, priors aligned to queries)."""
    jd_text = (jd_text or "").strip()
    sec = _light_split_sections(jd_text)
    queries: List[str] = []
    priors: List[float] = []

    # always include whole JD
    queries.append(jd_text)
    priors.append(1.0)

    # add key sections if present
    SEC_PRIOR = {"requirements": 2.0, "skills": 2.0, "responsibilities": 1.2, "about": 0.8, "summary": 1.0, "preferred": 1.0}
    for name in ["requirements", "skills", "responsibilities", "about"]:
        if name in sec and sec[name].strip() and sec[name].strip() not in queries:
            queries.append(sec[name].strip())
            priors.append(SEC_PRIOR.get(name, 1.0))

    # mine noun-phrases from req/skills (or entire JD if missing)
    base = " ".join([sec.get("requirements", ""), sec.get("skills", "")]).strip() or jd_text
    phrases = _top_np_phrases(base, k=6) 
    for p in phrases:
        if p not in queries:
            queries.append(p)
            priors.append(1.1)  # phrase queries slightly upweighted

    # trim / pad
    if len(queries) > max_q:
        queries, priors = queries[:max_q], priors[:max_q]
    while len(queries) < min_q:
        queries.append(jd_text)
        priors.append(1.0)

    priors = np.asarray(priors, dtype=np.float32)
    priors = priors / (priors.sum() or 1.0)
    return queries, priors

# -----------------------------
# Semantic scoring
# -----------------------------

YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")

def _len_cap(text: str, cap: int = 280) -> float:
    n = max(1, len(text.split()))
    return float(np.clip(n / float(cap), 0.2, 1.0))

def _sec_boost(sec: str) -> float:
    return {
        "skills": 2.0,
        "experience": 1.6,
        "projects": 1.3,
        "education": 0.9,
    }.get((sec or "other").lower(), 1.0)

def _recent_boost(text: str, years_window: int = 3) -> float:
    try:
        yrs = [int(y) for y in YEAR_RE.findall(text)]
        if not yrs: 
            return 1.0
        if max(yrs) >= datetime.now().year - years_window:
            return 1.1
    except Exception:
        pass
    return 1.0

def _mean_top_k(arr: np.ndarray, k: int = 2) -> np.ndarray:
    # arr shape: (nc, nq)
    if arr.shape[1] == 0:
        return np.zeros(arr.shape[0], dtype=np.float32)
    k = min(k, arr.shape[1])
    part = np.partition(arr, -k, axis=1)[:, -k:]
    return part.mean(axis=1)

def calculate_semantic_score_batched(jd_queries: List[str],
                                     jd_priors: np.ndarray,
                                     cv_chunks: List[dict],
                                     model: SentenceTransformer) -> Dict[str, any]:
    """Return dict with 'semantic' float and diagnostics."""
    if not cv_chunks:
        return {"semantic": 0.0, "chunk_scores": [], "section_sims": []}

    # Encode in batches (normalized)
    q_encoded = model.encode(
        [f"Represent this sentence for searching relevant passages: {q}" for q in jd_queries],
        normalize_embeddings=True, convert_to_numpy=True
    )  # (nq, d)

    chunk_texts = [c.get("text", "") for c in cv_chunks]
    c_encoded = model.encode(chunk_texts, normalize_embeddings=True, convert_to_numpy=True)  # (nc, d)

    # Similarity matrix
    S = np.matmul(c_encoded, q_encoded.T).astype(np.float32)  # (nc, nq)

    # Per-chunk pooling with JD priors
    mean_top2 = _mean_top_k(S, k=2)                           # (nc,)
    prior_wmean = (S * jd_priors[None, :]).sum(axis=1)        # (nc,)
    base_chunk = 0.3 * mean_top2 + 0.7 * prior_wmean          # (nc,)

    # Quality **weights** (no direct shrinking of similarity)
    len_caps   = np.array([_len_cap(t) for t in chunk_texts], dtype=np.float32)
    sec_boosts = np.array([_sec_boost(c.get("section")) for c in cv_chunks], dtype=np.float32)
    rec_boosts = np.array([_recent_boost(t) for t in chunk_texts], dtype=np.float32)
    weights    = len_caps * sec_boosts * rec_boosts           # (nc,)

    # Weighted average + best (from base similarities)
    wavg = float((base_chunk * weights).sum() / (weights.sum() or 1.0))
    best = float(base_chunk.max() if base_chunk.size else 0.0)
    semantic = 0.7 * wavg + 0.3 * best

    # Bound to [0,1]
    semantic = float(np.clip(semantic, 0.0, 1.0))

    # Diagnostics: sample some per-section sims
    section_sims = {
        "mean_top2_head": float(mean_top2.mean()),
        "prior_wmean_head": float(prior_wmean.mean()),
        "len_cap_mean": float(len_caps.mean()),
        "sec_boost_mean": float(sec_boosts.mean()),
        "rec_boost_mean": float(rec_boosts.mean()),
    }

    # Per-chunk scores for diagnostics: combine base similarity with quality weights
    try:
        chunk_scores = (base_chunk * weights).astype(np.float32)
    except Exception:
        # Fallback: use base_chunk
        chunk_scores = base_chunk.astype(np.float32)

    return {
        "semantic": semantic,
        "chunk_scores": chunk_scores.tolist(),
        "section_sims": section_sims
    }

# -----------------------------
# Public entry (keeps API shape)
# -----------------------------

def improved_rank_cv(jd_text: str, cv_path: str, model_name: str = "bge-base-en-v1.5") -> Tuple[float, dict, str]:
    """
    Returns (score_0_1, cv_data, explanation)
    """
    try:
        model = get_model(model_name)

        # Parse CV with ~280/60 chunks
        cv_data = P.parse_cv_file(cv_path, chunk_max_words=280, chunk_overlap_words=60)
        chunks = cv_data.get("chunks", [])

        if not chunks:
            full_text = cv_data.get("full_text", "")
            if full_text:
                chunks = [{"text": full_text, "section": "full_text"}]
            else:
                return 0.0, cv_data, "No text content found in CV"

        # Build JD queries (6–12) with priors
        jd_queries, jd_priors = build_jd_queries(jd_text, min_q=6, max_q=12)

        # Batched semantic scoring
        ss = calculate_semantic_score_batched(jd_queries, jd_priors, chunks, model)
        score = ss["semantic"]

        # Human explanation (short)
        top3 = sorted(ss["chunk_scores"], reverse=True)[:3]
        explanation = (
            f"Semantic-only scoring with BGE prompt. "
            f"JD queries={len(jd_queries)}; CV chunks={len(chunks)}; "
            f"chunk top-3={', '.join(f'{x:.3f}' for x in top3)}. "
            f"Final semantic={score:.3f}."
        )

        return score, cv_data, explanation

    except Exception as e:
        return 0.0, {}, f"Error in ranking: {str(e)}"

# (Optional) direct test
if __name__ == "__main__":
    print("Run tests via your pytest files or call improved_rank_cv() directly.")
