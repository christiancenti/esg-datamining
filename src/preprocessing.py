"""
EcoScan: Modulo di Preprocessing per Mining di Sostenibilità Potenziato dall'IA
Gestisce l'estrazione di testo da PDF, la pulizia rigorosa e la strutturazione in Markdown prima dell'ingestione da parte dell'LLM.
"""

import re
import io
import pdfplumber
import nltk
from nltk.corpus import stopwords
from typing import Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parole chiave per identificare i paragrafi rilevanti ESG (Filtraggio Rilevanza Contenuto)
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

# Pattern per la Rimozione del Rumore di Layout
NOISE_PATTERNS = [
    r"^\d+\s*$",                  # Numeri di pagina (es. "12")
    r"^page\s+\d+$",              # Numeri di pagina (es. "Page 12")
    r"^\s*report\s*$",            # Intestazioni generiche
    r"confidential",              # Marker di confidenzialità
    r"all rights reserved",       # Copyright
    r"\(cid:\d+\)",               # Artefatti di codifica PDF
    r"www\.[a-z0-9-]+\.[a-z]+",   # URL (spesso nei piè di pagina)
]

# Carica Stopwords (Inglese + Italiano per robustezza)
nltk_stopwords = set(stopwords.words('english')).union(set(stopwords.words('italian')))

# Stopwords Aziendali/Reporting personalizzate per rimuovere rumore generico
CORPORATE_STOPWORDS = {
    "group", "report", "fiscal", "year", "annual", "continued", "commitment", 
    "company", "strategy", "management", "performance", "target", "approach",
    "page", "total", "new", "business", "data", "reporting", "also", "within"
}

STOP_WORDS = nltk_stopwords.union(CORPORATE_STOPWORDS)

# ============================================================================
# UTILITIES
# ============================================================================

def count_tokens(text: str) -> int:
    """
    Stima il conteggio dei token usando il tokenizer di parole di nltk.
    Veloce e sufficiente per il confronto.
    """
    words = nltk.word_tokenize(text)
    return int(len(words))


def is_relevant(text: str) -> bool:
    """Controlla se il paragrafo contiene parole chiave ESG rilevanti."""
    text_lower = text.lower()
    return any(re.search(k, text_lower) for k in ESG_KEYWORDS)


def is_noise(text: str) -> bool:
    """Controlla se la riga corrisponde a pattern di rumore (intestazioni, piè di pagina, ecc.)."""
    text_lower = text.strip().lower()
    if len(text_lower) < 5: # Salta frammenti molto brevi a meno che non sembrino intestazioni
        return True
        
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def remove_stopwords(text: str) -> str:
    """
    Rimuove le stop word comuni usando NLTK per ridurre il rumore.
    Mantiene la punteggiatura attaccata alle parole (split semplice) per preservare la struttura della frase.
    """
    words = nltk.word_tokenize(text)
    # Keep words that are NOT in stop words (case insensitive check)
    filtered = [w for w in words if w.lower() not in STOP_WORDS]
    return " ".join(filtered)


def clean_text(raw_text: str) -> str:
    """
    Applica tecniche di pulizia dati standard al testo grezzo.
    1. Rimuove rumore di layout valido.
    2. Normalizza gli spazi bianchi.
    3. Unisce le righe in paragrafi per un migliore filtraggio semantico.
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
        
        # Opzionale: Applica Rimozione Stopword qui o parzialmente?
        # L'applicazione a livello di riga aiuta la riduzione dei token.
        line = remove_stopwords(line)
        
        cleaned_lines.append(line)
        
    # Seconda passata: Unisce le righe in paragrafi
    # euristica: se la riga è breve (<60 caratteri) e finisce con punteggiatura, è probabilmente la fine di un paragrafo.
    # altrimenti, unisci con spazio.
    merged_paragraphs = []
    current_para = []
    
    for line in cleaned_lines:
        current_para.append(line)
        
        # Controlla se il paragrafo dovrebbe finire
        # Finisce con . ! ? o corrisponde a un pattern di Intestazione (Breve & Maiuscolo)
        is_end_char = line.endswith(('.', '!', '?', ':'))
        is_header = line.isupper() and len(line) < 100
        
        if is_end_char or is_header:
            merged_paragraphs.append(" ".join(current_para))
            current_para = []
            
    if current_para:
        merged_paragraphs.append(" ".join(current_para))
        
    return "\n\n".join(merged_paragraphs)


def extract_top_keywords(text_content: str, top_n: int = 10) -> List[str]:
    """
    Estrae le Prime N parole chiave usando TF-IDF (Frequenza del Termine - Frequenza Inversa del Documento).
    Tratta i paragrafi come documenti per identificare parole che sono localmente significative 
    piuttosto che solo globalmente frequenti.
    """
    if not text_content:
        return []
    
    # 1. Dividi il testo in paragrafi (i nostri "documenti")
    paragraphs = [p.strip() for p in text_content.split('\n\n') if len(p.split()) > 5]
    
    if len(paragraphs) < 2:
        return []

    try:
        # 2. Calcola TF-IDF
        # Usa stopwords Inglesi + Italiane per robustezza
        # ngram_range=(1,2) cattura "renewable energy" o "cambiamento climatico"
        vectorizer = TfidfVectorizer(
            stop_words=list(STOP_WORDS), 
            ngram_range=(1, 2), 
            max_df=0.95, # Limite superiore rilassato
            min_df=1     # Limite inferiore rilassato per lavorare su testi brevi/report di esempio
        )
        
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        
        # 3. Somma i punteggi TF-IDF per ogni termine attraverso tutti i paragrafi
        # Questo ci dà l' "importanza complessiva" del termine nel report
        sum_scores = tfidf_matrix.sum(axis=0) 
        
        # 4. Mappa termini ai punteggi
        features = vectorizer.get_feature_names_out()
        term_scores = [(features[i], sum_scores[0, i]) for i in range(len(features))]
        
        # 5. Ordina per punteggio decrescente
        sorted_terms = sorted(term_scores, key=lambda x: x[1], reverse=True)
        
        return [term for term, score in sorted_terms[:top_n]]
        
    except ValueError:
        # Può accadere se il vocabolario è vuoto dopo il filtraggio
        return []






def to_markdown(text: str) -> str:
    """
    Trasforma il testo pulito in struttura Markdown.
    Euristiche:
    - RIGHE MAIUSCOLE -> ## Intestazioni
    - Righe che iniziano con punti elenco -> Elementi lista
    """
    lines = text.split('\n\n') # Split by paragraph
    md_lines = []
    
    for line in lines:
        # Euristica: Breve, Maiuscolo = Intestazione
        if line.isupper() and len(line) < 100:
            md_lines.append(f"## {line.title()}")
        # Euristica: Inizia con carattere punto elenco
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
    Metrica Metodologica: Densità CSR
    Rapporto di frasi/contenuto ESG-rilevante vs Contenuto Totale.
    Semplice proxy operativo: Conteggio Caratteri Rilevanti / Conteggio Caratteri Totali
    (Più robusto del semplice conteggio parole chiave distinte per questo ambito).
    """
    if not cleaned_text:
        return 0.0
    return round(len(relevant_text) / len(cleaned_text), 4)

def process_pdf_pipeline(file_bytes: bytes) -> Tuple[str, str, dict]:
    """
    Esegui la Pipeline completa di Data Mining:
    PDF -> Testo -> Pulizia -> Filtro -> Markdown
    
    Ritorna:
        processed_text: Stringa Markdown finale
        raw_text: Testo estratto originale (per confronto)
        metrics: Dizionario con statistiche efficienza token e metriche metodologiche
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
    
    # 2. Pulizia & Normalizzazione
    cleaned_text = clean_text(raw_text)
    cleaned_tokens = count_tokens(cleaned_text) # Token totali post-cleaning
    
    # 3. Filtraggio Rilevanza Contenuto (Livello Paragrafo)
    # Dividiamo per doppia nuova riga per approssimare i paragrafi
    paragraphs = cleaned_text.split('\n\n')
    relevant_paragraphs = [p for p in paragraphs if is_relevant(p)]
    
    # Fallback: Se il filtro stretto rimuove troppo (>95%), mantieni pulito originale
    if len(relevant_paragraphs) < len(paragraphs) * 0.05:
        final_text_content = cleaned_text
    else:
        final_text_content = "\n\n".join(relevant_paragraphs)
        
    # Metrica Metodologica: Densità CSR (Basata su Frasi)
    csr_density = calculate_csr_density(cleaned_text, final_text_content)

    # 4. Markdown Structuring
    markdown_text = to_markdown(final_text_content)
    
    # Metrics: Final (ESG-relevant tokens)
    final_tokens = count_tokens(markdown_text)
    
    # Calculate Reductions
    reduction = 0.0
    if initial_tokens > 0:
        reduction = ((initial_tokens - final_tokens) / initial_tokens) * 100
    
    # Metrica Metodologica: Proxy di Sinteticità (Basato su Token)
    # (Token ESG-rilevanti) / (Token totali post-cleaning)
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
