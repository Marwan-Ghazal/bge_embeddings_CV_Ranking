#!/usr/bin/env python3
"""
Improved CV ranking system that fixes the parsing and scoring issues.
Addresses the problem where graduate AI developers score lower than sophomore students.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import parsing as P

# Model cache
_MODELS = {}

# Map friendly local names to Hugging Face repo IDs for fallback
REMOTE_MODEL_MAP = {
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    
}

def get_model(name: str = "bge-base-en-v1.5") -> SentenceTransformer:
    """Get or load a SentenceTransformer model.

    Strategy:
    1) If a local directory exists for `name` (or common local aliases), load it.
    2) Otherwise, fall back to Hugging Face hub using a mapped repo id.
    """
    if name in _MODELS:
        return _MODELS[name]

    # 1) Try local paths first
    candidate_paths = [
        name,
        name.rstrip("/") + "/",
    ]
    # Add common local aliases for convenience
    if name.startswith("bge-base-en-v1.5"):
        candidate_paths.extend(["bge-base-en-v1.5", "bge-base-en-v1.5/"])
   
    found_local = None
    for path in candidate_paths:
        try:
            if os.path.exists(path) and os.path.isdir(path):
                found_local = path
                break
        except Exception:
            continue

    if found_local:
        _MODELS[name] = SentenceTransformer(found_local)
        print(f"✅ Loaded local model: {found_local}")
        return _MODELS[name]

    # 2) Fall back to Hugging Face repo
    repo_id = REMOTE_MODEL_MAP.get(name, name)
    try:
        _MODELS[name] = SentenceTransformer(repo_id)
        print(f"🌐 Loaded remote model from HF: {repo_id}")
        return _MODELS[name]
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{name}'. Tried local paths {candidate_paths} and remote repo '{repo_id}'. "
            f"Error: {e}"
        )

def improved_parse_cv(path: str) -> dict:
    """
    Improved CV parsing that handles cases where section detection fails.
    Falls back to intelligent text chunking if sections can't be identified.
    """
    try:
        # Try standard parsing first
        cv_data = P.parse_cv_file(path, chunk_max_words=900, chunk_overlap_words=200)
        
        # Check if parsing was successful
        sections = cv_data.get('sections', {})
        
        # If only 'other' section or parsing failed, use fallback approach
        if len(sections) <= 1 or 'other' in sections and len(sections) == 1:
            print(f"⚠️  Section parsing failed for {path}, using fallback approach")
            
            # Extract full text
            full_text = cv_data.get('full_text', '')
            if not full_text:
                return cv_data
            
            # Use intelligent chunking based on content
            chunks = []
            
            # Split by common CV patterns
            lines = full_text.split('\n')
            current_chunk = []
            current_word_count = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line looks like a section header
                is_header = any(keyword in line.lower() for keyword in [
                    'education', 'experience', 'skills', 'projects', 'summary', 
                    'objective', 'work', 'employment', 'technical', 'certifications'
                ])
                
                # Start new chunk if we hit a header or current chunk is getting too long
                if is_header or current_word_count > 800:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text.split()) > 50:  # Only add substantial chunks
                            chunks.append({
                                'text': chunk_text,
                                'section': 'content',
                                'start_word': 0,
                                'end_word': len(chunk_text.split()),
                                'token_len_est': int(len(chunk_text.split()) * 0.75)
                            })
                    current_chunk = [line]
                    current_word_count = len(line.split())
                else:
                    current_chunk.append(line)
                    current_word_count += len(line.split())
            
            # Add final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) > 50:
                    chunks.append({
                        'text': chunk_text,
                        'section': 'content',
                        'start_word': 0,
                        'end_word': len(chunk_text.split()),
                        'token_len_est': int(len(chunk_text.split()) * 0.75)
                    })
            
            # Update CV data with improved chunks
            cv_data['chunks'] = chunks
            cv_data['sections'] = {'improved_content': 'Content parsed with fallback method'}
            
            print(f"✅ Fallback parsing created {len(chunks)} chunks")
        
        return cv_data
        
    except Exception as e:
        print(f"❌ Error in improved parsing: {e}")
        # Return basic structure if all else fails
        return {
            'cv_id': os.path.basename(path),
            'source_path': path,
            'chunks': [],
            'sections': {},
            'full_text': '',
            'contact': {'emails': [], 'phones': [], 'links': []}
        }

def calculate_semantic_score(jd_embedding: np.ndarray, cv_chunks: List[dict], model: SentenceTransformer) -> float:
    """
    Calculate semantic similarity score with improved weighting.
    Considers chunk quality and relevance.
    """
    if not cv_chunks:
        return 0.0
    
    chunk_scores = []
    chunk_weights = []
    
    for chunk in cv_chunks:
        chunk_text = chunk.get('text', '').strip()
        if not chunk_text or len(chunk_text.split()) < 10:
            continue
        
        # Encode chunk
        chunk_emb = model.encode([chunk_text], normalize_embeddings=True)[0]
        similarity = np.dot(chunk_emb, jd_embedding)
        
        # Weight chunks by quality (longer, more structured chunks get higher weight)
        word_count = len(chunk_text.split())
        quality_weight = min(word_count / 100.0, 2.0)  # Cap at 2.0x weight
        
        chunk_scores.append(similarity)
        chunk_weights.append(quality_weight)
    
    if not chunk_scores:
        return 0.0
    
    # Calculate weighted average
    total_weight = sum(chunk_weights)
    if total_weight == 0:
        return max(chunk_scores) if chunk_scores else 0.0
    
    weighted_score = sum(score * weight for score, weight in zip(chunk_scores, chunk_weights)) / total_weight
    
    # Also consider the best chunk score
    best_score = max(chunk_scores)
    
    # Combine weighted average with best score (70% weighted, 30% best)
    final_score = 0.7 * weighted_score + 0.3 * best_score
    
    return final_score

def improved_rank_cv(
    jd_text: str,
    cv_path: str,
    model_name: str = "bge-base-en-v1.5"
) -> Tuple[float, dict, str]:
    """
    Improved CV ranking that addresses parsing and scoring issues.
    
    Returns:
        (score, cv_data, explanation)
    """
    try:
        # Load model
        model = get_model(model_name)
        
        # Prepare query with better instruction
        query = f"Represent this job description for searching relevant CVs: {jd_text}"
        jd_embedding = model.encode([query], normalize_embeddings=True)[0]
        
        # Parse CV with improved method
        cv_data = improved_parse_cv(cv_path)
        
        # Extract chunks
        chunks = cv_data.get('chunks', [])
        
        if not chunks:
            # Fallback: use full text as single chunk
            full_text = cv_data.get('full_text', '')
            if full_text:
                chunks = [{'text': full_text, 'section': 'full_text'}]
            else:
                return 0.0, cv_data, "No text content found in CV"
        
        # Calculate semantic score
        score = calculate_semantic_score(jd_embedding, chunks, model)
        
        # Generate explanation
        explanation = f"CV parsed into {len(chunks)} chunks. "
        if 'improved_content' in cv_data.get('sections', {}):
            explanation += "Used fallback parsing due to section detection failure. "
        
        explanation += f"Final score: {score:.3f} ({score*100:.1f}%)"
        
        return score, cv_data, explanation
        
    except Exception as e:
        return 0.0, {}, f"Error in ranking: {str(e)}"

def test_improved_ranking():
    """Test the improved ranking system."""
    print("🧪 Testing Improved CV Ranking System...")
    print("=" * 60)
    
    # Job description
    jd_text = """Job Title: AI Developer
Location: Cairo, Egypt (Hybrid)
Employment Type: Full-time

About the Role
We are seeking a passionate and skilled AI Developer to join our technology team. You will be responsible for designing, developing, and deploying AI-driven solutions that address complex business challenges. The ideal candidate has strong problem-solving skills, experience with machine learning frameworks, and the ability to work collaboratively in a fast-paced environment.

Key Responsibilities
Design, implement, and optimize machine learning models for various applications.
Preprocess and analyze large datasets to extract actionable insights.
Collaborate with data scientists, backend developers, and product managers to integrate AI models into production systems.
Conduct research and experiments to improve model accuracy, efficiency, and scalability.
Develop APIs and interfaces to expose AI model functionality to other services.
Document processes, models, and code to ensure maintainability and reproducibility.

Requirements
Bachelor's or Master's degree in Computer Science, Artificial Intelligence, Data Science, or a related field.
Strong proficiency in Python and libraries such as NumPy, Pandas, Scikit-learn, and TensorFlow/PyTorch.
Experience with Natural Language Processing (NLP) and/or Computer Vision.
Understanding of cloud platforms (AWS, Azure, GCP) for model deployment.
Familiarity with MLOps tools and workflows.
Strong problem-solving and analytical thinking skills.
Excellent communication and teamwork abilities.

Preferred Qualifications
Experience with large language models (LLMs) and prompt engineering.
Knowledge of vector databases and embedding models.
Contributions to open-source AI projects.
Understanding of distributed computing and GPU optimization."""
    
    # Test CVs
    cv_paths = [
        "data/cvs/Resume-MarwanGhazal.pdf",      # Sophomore AI student
        "data/cvs/Mohamed_Osama_CV.pdf"          # Graduate AI developer
    ]
    
    cv_names = [
        "Resume-MarwanGhazal (Sophomore AI Student)",
        "Mohamed_Osama_CV (Graduate AI Developer)"
    ]
    
    results = []
    
    for cv_path, cv_name in zip(cv_paths, cv_names):
        print(f"\n📋 Testing: {cv_name}")
        print(f"📁 Path: {cv_path}")
        
        if not os.path.exists(cv_path):
            print(f"❌ CV file not found: {cv_path}")
            continue
        
        try:
            score, cv_data, explanation = improved_rank_cv(jd_text, cv_path)
            
            print(f"✅ Score: {score:.3f} ({score*100:.1f}%)")
            print(f"📝 Explanation: {explanation}")
            
            # Show parsing results
            sections = cv_data.get('sections', {})
            chunks = cv_data.get('chunks', [])
            print(f"   - Sections: {list(sections.keys())}")
            print(f"   - Chunks: {len(chunks)}")
            
            results.append({
                'name': cv_name,
                'score': score,
                'explanation': explanation
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 IMPROVED RANKING RESULTS")
    print("=" * 60)
    
    if len(results) == 2:
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"🥇 Highest Score: {results[0]['name']} - {results[0]['score']:.3f} ({results[0]['score']*100:.1f}%)")
        print(f"🥈 Lower Score: {results[1]['name']} - {results[1]['score']:.3f} ({results[1]['score']*100:.1f}%)")
        
        score_diff = results[0]['score'] - results[1]['score']
        print(f"📈 Score Difference: {score_diff:.3f} ({score_diff*100:.1f}%)")
        
        # Check if ranking makes sense now
        if "Graduate AI Developer" in results[0]['name'] and "Sophomore AI Student" in results[1]['name']:
            print("✅ IMPROVED: Ranking is now CORRECT!")
        elif "Sophomore AI Student" in results[0]['name'] and "Graduate AI Developer" in results[1]['name']:
            print("❌ Still incorrect ranking - further investigation needed")
        else:
            print("⚠️  Unexpected ranking order")
    
    return results

if __name__ == "__main__":
    test_improved_ranking()
