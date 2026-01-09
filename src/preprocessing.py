"""
EcoScan: AI-Powered Sustainability Mining - Preprocessing Module
Handles PDF text extraction, rigorous cleaning, and Markdown structuring before LLM ingestion.
Aligned with PRD v1.0.
"""

import re
import io
import pdfplumber
from typing import Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

# Keywords to identify ESG-relevant paragraphs (Content Relevance Filtering)
ESG_KEYWORDS = [
    # Environment
    r"emission", r"ghg", r"scope 1", r"scope 2", r"scope 3", r"energy", 
    r"renewable", r"carbon", r"climate", r"water", r"waste", r"biodiversity",
    r"tco2", r"mwh", r"gj",
    
    # Social
    r"safety", r"injury", r"trir", r"incident", r"fatality", r"workforce",
    r"employee", r"diversity", r"women", r"gender", r"training", r"human rights",
    r"community",
    
    # Governance
    r"governance", r"board", r"supplier", r"supply chain", r"traceability",
    r"code of conduct", r"corruption", r"audit", r"compliance", r"risk",
    r"materiality", r"stakeholder"
]

# Patterns for Layout Noise Removal
NOISE_PATTERNS = [
    r"^\d+\s*$",                  # Page numbers (e.g., "12")
    r"^page\s+\d+$",              # Page numbers (e.g., "Page 12")
    r"^\s*report\s*$",            # Generic headers
    r"confidential",              # Confidentiality markers
    r"all rights reserved",       # Copyright
    r"\(cid:\d+\)",               # PDF encoding artifacts
    r"www\.[a-z0-9-]+\.[a-z]+",   # URLs (often in footers)
]

# ============================================================================
# UTILITIES
# ============================================================================

def count_tokens(text: str) -> int:
    """
    Estimate token count using word-based approximation (1 word ~ 1.3 tokens).
    Fast and sufficient for comparison.
    """
    words = len(text.split())
    return int(words * 1.3)


def is_relevant(text: str) -> bool:
    """Check if paragraph contains relevant ESG keywords."""
    text_lower = text.lower()
    return any(re.search(k, text_lower) for k in ESG_KEYWORDS)


def is_noise(text: str) -> bool:
    """Check if line matches noise patterns (headers, footers, etc.)."""
    text_lower = text.strip().lower()
    if len(text_lower) < 5: # Skip very short snippets unless they look like headers
        return True
        
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def clean_text(raw_text: str) -> str:
    """
    Apply standard data cleaning techniques to raw text (PRD Section 6).
    1. Remove valid layout noise.
    2. Normalize whitespace.
    3. Merge lines into paragraphs for better semantic filtering.
    """
    lines = raw_text.split('\n')
    cleaned_lines = []
    
    # First pass: Basic cleaning
    for line in lines:
        line = line.strip()
        if not line: continue
        if is_noise(line): continue
        line = re.sub(r"\(cid:\d+\)", "", line)
        line = re.sub(r"\s+", " ", line)
        cleaned_lines.append(line)
        
    # Second pass: Merge lines into paragraphs
    # heuristic: if line is short (<60 chars) and ends with punctuation, it's likely a paragraph end.
    # otherwise, join with space.
    merged_paragraphs = []
    current_para = []
    
    for line in cleaned_lines:
        current_para.append(line)
        
        # Check if paragraph should end
        # Ends with . ! ? or matches a Header pattern (Short & Upper)
        is_end_char = line.endswith(('.', '!', '?', ':'))
        is_header = line.isupper() and len(line) < 100
        
        if is_end_char or is_header:
            merged_paragraphs.append(" ".join(current_para))
            current_para = []
            
    if current_para:
        merged_paragraphs.append(" ".join(current_para))
        
    return "\n\n".join(merged_paragraphs)

# ... inside process_pdf_pipeline ... (no changes needed if clean_text returns paragraphs separated by \n\n)



def to_markdown(text: str) -> str:
    """
    Transform cleaned text into Markdown structure (PRD Section 8).
    Heuristics:
    - UPPERCASE LINES -> ## Headers
    - Lines starting with bullet points -> List items
    """
    lines = text.split('\n\n') # Split by paragraph
    md_lines = []
    
    for line in lines:
        # Heuristic: Short, Uppercase = Header
        if line.isupper() and len(line) < 100:
            md_lines.append(f"## {line.title()}")
        # Heuristic: Starts with bullet char
        elif line.strip().startswith(('•', '-', '–')):
            md_lines.append(f"- {line.strip().lstrip('•-– ')}")
        else:
            md_lines.append(line)
            
    return "\n\n".join(md_lines)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def calculate_csr_density(cleaned_text: str, relevant_text: str) -> float:
    """
    Methodological Metric: CSR Density
    Ratio of ESG-relevant sentences/content vs Total content.
    Simple operational proxy: Relevant Char Count / Total Char Count
    (More robust than simple distinct keyword counting for this scope).
    """
    if not cleaned_text:
        return 0.0
    return round(len(relevant_text) / len(cleaned_text), 4)

def process_pdf_pipeline(file_bytes: bytes) -> Tuple[str, str, dict]:
    """
    Execute the full Data Mining Pipeline:
    PDF -> Text -> Clean -> Filter -> Markdown
    
    Returns:
        processed_text: Final Markdown string
        raw_text: Original extracted text (for comparison)
        metrics: Dictionary with token efficiency stats and methodological metrics
    """
    raw_text_pages = []
    
    # 1. Text Extraction (pdfplumber)
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            # pdfplumber extracts better structural text
            text = page.extract_text()
            if text:
                raw_text_pages.append(text)
                
    raw_text = "\n".join(raw_text_pages)
    
    # Metrics: Initial
    initial_tokens = count_tokens(raw_text)
    
    # 2. Cleaning & Normalization
    cleaned_text = clean_text(raw_text)
    cleaned_tokens = count_tokens(cleaned_text) # Token totali post-cleaning
    
    # 3. Content Relevance Filtering (Paragraph level)
    # We split by double newline to approximate paragraphs
    paragraphs = cleaned_text.split('\n\n')
    relevant_paragraphs = [p for p in paragraphs if is_relevant(p)]
    
    # Fallback: If strict filtering removes too much (>95%), keep original cleaned
    if len(relevant_paragraphs) < len(paragraphs) * 0.05:
        final_text_content = cleaned_text
    else:
        final_text_content = "\n\n".join(relevant_paragraphs)
        
    # Methodological Metric: CSR Density (Sentence-based)
    csr_density = calculate_csr_density(cleaned_text, final_text_content)

    # 4. Markdown Structuring
    markdown_text = to_markdown(final_text_content)
    
    # Metrics: Final (ESG-relevant tokens)
    final_tokens = count_tokens(markdown_text)
    
    # Calculate Reductions
    reduction = 0.0
    if initial_tokens > 0:
        reduction = ((initial_tokens - final_tokens) / initial_tokens) * 100
    
    # Methodological Metric: Conciseness Proxy (Token-based)
    # (Token ESG-relevant) / (Token totali post-cleaning)
    conciseness = 0.0
    if cleaned_tokens > 0:
        conciseness = round(final_tokens / cleaned_tokens, 4)
        
    metrics = {
        "initial_tokens": initial_tokens,
        "final_tokens": final_tokens,
        "reduction_pct": round(reduction, 2),
        "csr_density": csr_density,
        "conciseness": conciseness
    }
    
    return markdown_text, raw_text, metrics
