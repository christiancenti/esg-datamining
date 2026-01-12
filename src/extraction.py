import os
from typing import Optional, Callable, Tuple
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

from src.models import ESGReport
from src.preprocessing import process_pdf_pipeline, extract_top_keywords

# Load environment variables
load_dotenv()

# ============================================================================
# PROMPT DEFINITIONS
# ============================================================================

ESG_EXTRACTION_PROMPT_SYSTEM = """
Sei un esperto analista ESG per una società di consulenza di alto livello.
Il tuo compito è estrarre dati ESG strutturati dai report di sostenibilità aziendali.

Riceverai il testo di un report (già pre-elaborato e pulito).
Devi estrarre 6 specifici Indicatori Chiave di Prestazione (KPI) e mapparli allo schema JSON fornito.

## DEFINIZIONI KPI:
1. **Intensità GHG (E1)**: (Ambientale) Emissioni Scope 1+2 normalizzate per fatturato (es. tCO2e/€M).
2. **Quota Energia Rinnovabile (E2)**: (Ambientale) Percentuale di elettricità da fonti rinnovabili.
3. **TRIR (S1)**: (Sociale) Tasso Totale di Incidenti Registrabili (Metrica di sicurezza).
4. **Donne in Leadership (S2)**: (Sociale) Percentuale di donne in ruoli manageriali/esecutivi.
5. **Punteggio ESG Fornitori (G1)**: (Governance) Punteggio medio o copertura della valutazione dei fornitori (es. EcoVadis, Sedex).
6. **Tracciabilità Supply Chain (G2)**: (Governance) Percentuale di materie prime tracciabili fino alla fonte.

## REGOLE CRITICHE:
- **FORMATTAZIONE**:
    - `value`: DEVE essere una stringa numerica/percentuale pulita (es. "550", "22.5", "3.6"). NON includere testo qui.
    - `unit`: Estrai l'unità separatamente (es. "tCO2e/€M", "%", "TRIR", "Score").
    - `trend`: ESTRAI SOLO se esplicitamente dichiarato nel testo (es. "diminuito del 5%"). NON CALCOLARLO.
- **SINTETICITÀ**: I campi di testo (standard) devono essere BREVI.
- **Anno Fiscale**: Estrai l'anno fiscale (es. "FY 2023", "2023").
- **Nome Azienda**: Estrai il nome esatto dell'azienda.
- **Trend**: NON CALCOLARE I TREND tra gli anni. Estrai solo dichiarazioni esplicite.
- Se una metrica NON viene trovata, lasciala come null (non allucinare).
"""

# ============================================================================
# GENAI CONFIGURATION
# ============================================================================

def get_genai_client():
    api_key = None
    
    # Priority 1: Streamlit Secrets (for Cloud Deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except (ImportError, FileNotFoundError):
        pass # Streamlit not installed or secrets not found

    # Priority 2: Environment Variables (Local .env)
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY non trovata in st.secrets o nelle variabili d'ambiente.")
    
    return genai.Client(api_key=api_key)

# ============================================================================
# MAIN AGENT FUNCTION
# ============================================================================

def analyze_structure(uploaded_file, log_callback: Optional[Callable[[str], None]] = None) -> Tuple[str, str, dict]:
    """Step 1: Preprocessing & Methodological Analysis (No LLM)."""
    def log(msg):
        if log_callback: log_callback(msg)
        print(f"[Preprocessing] {msg}")

    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    
    log(f"🚀 Pre-elaborazione: {filename}")
    
    
    if filename.endswith('.pdf'):
        processed_text, raw_text, metrics = process_pdf_pipeline(file_bytes)
        
        # Calculate TF-IDF Top Keywords
        log("   🧮 Calcolo Temi Principali TF-IDF...")
        top_kwd = extract_top_keywords(processed_text, top_n=8)
        metrics["top_keywords"] = top_kwd
        
        log(f"   📊 Efficienza Token: ▼ {metrics['reduction_pct']}%")
        return processed_text, raw_text, metrics
        
    return "", "", {}


def extract_kpis_with_llm(processed_text: str, token_metrics: dict, log_callback: Optional[Callable[[str], None]] = None) -> ESGReport:
    """Step 2: LLM Extraction using pre-processed text."""
    def log(msg):
        if log_callback: log_callback(msg)
        print(f"[LLM-Agent] {msg}")

    client = get_genai_client()
    
    try:
        log("   🤖 Estrazione KPI ESG in corso (Google Native SDK)...")

        # Define Deterministic Tool
        def calculate_kpi(numerator: float, denominator: float) -> float:
            """Calculates a KPI intensity or ratio (numerator / denominator)."""
            log(f"      🧮 Tool Attivato: {numerator} / {denominator}")
            if denominator == 0: return 0.0
            res = round(numerator / denominator, 4)
            log(f"      ↳ Result: {res}")
            return res

        # Update Prompt
        TOOL_PROMPT_ADDITION = """
## REGOLE DI CALCOLO:
- Se trovi numeri grezzi (es. Emissioni Totali = 1000, Ricavi = 50), NON calcolare l'intensità da solo.
- USA lo strumento `calculate_kpi` per calcolare il risultato in modo deterministico.
- Es. chiama calculate_kpi(1000, 50) -> 20.0

## REGOLA DI STRICT GROUNDING:
- ESTRAI SOLO DATI ESPLICITAMENTE PRESENTI NEL TESTO.
- NON INFERIRE O INDOVINARE VALORI MANCANTI.
- Se un valore manca, restituisci null.
"""
        
        # Use Chat interface for multi-turn tool execution
        chat = client.chats.create(
            model="gemini-3-flash-preview", 
            config={
                'tools': [calculate_kpi], 
                'response_mime_type': 'application/json',
                'response_schema': ESGReport,
            }
        )
        
        # Send initial message
        response = chat.send_message(
            f"{ESG_EXTRACTION_PROMPT_SYSTEM}\n{TOOL_PROMPT_ADDITION}\n\nEcco il contenuto del report:\n\n{processed_text}"
        )

        # Handle Function Calls
        for _ in range(5): # Max 5 turns
            part =  response.candidates[0].content.parts[0]
            if part.function_call:
                fc = part.function_call
                fn_name = fc.name
                args = fc.args
                
                if fn_name == 'calculate_kpi':
                    result = calculate_kpi(args['numerator'], args['denominator'])
                    response = chat.send_message(
                        genai.types.Content(
                            parts=[genai.types.Part(
                                function_response=genai.types.FunctionResponse(
                                    name=fn_name,
                                    response={'result': result}
                                )
                            )]
                        )
                    )
                    continue
            else:
                break
        
        # Parse output
        if response.parsed:
            report = response.parsed
        elif response.text:
            report = ESGReport.model_validate_json(response.text)
        else:
            log("⚠️ Nessun dato restituito dal modello.")
            return ESGReport(extraction_confidence=0.0)

        # INJECT METHODOLOGICAL METRICS
        report.initial_tokens = token_metrics["initial_tokens"]
        report.final_tokens = token_metrics["final_tokens"]
        report.token_reduction_pct = token_metrics["reduction_pct"]
        report.csr_density = token_metrics.get("csr_density", 0.0)
        report.conciseness_proxy = token_metrics.get("conciseness", 0.0)
        
        report.extraction_confidence = 1.0 

        # --- GENERATE RECAP ---
        try:
            log("   📝 Generazione Executive Summary...")
            recap_text = generate_data_recap(client, report)
            report.recap = recap_text
        except Exception as e:
                log(f"⚠️ Recap generation failed: {str(e)}")
        
        log("✅ Estrazione completata!")
        return report

    except Exception as e:
        log(f"❌ Errore durante l'estrazione LLM: {str(e)}")
        import traceback
        traceback.print_exc()
        return ESGReport(extraction_confidence=0.0)





# ============================================================================
# RECAP GENERATOR
# ============================================================================

def generate_data_recap(client: genai.Client, report: ESGReport) -> str:
    """
    Generate a strictly data-grounded executive summary using standard text generation.
    """
    
    RECAP_PROMPT = """Sei un Analista Dati ESG.
    Il tuo compito è scrivere una sintesi esecutiva BREVE e NEUTRALE (max 4-5 frasi) basata SOLO sui dati strutturati forniti.
    
    LINEE GUIDA:
    - Riassumi quali aree ESG (Ambiente, Sociale, Governance) hanno copertura dati.
    - Menziona esplicitamente metriche mancanti o incomplete.
    - Menziona trend positivi/negativi se presenti nei dati.
    - Fai riferimento alle Metriche di Qualità dei Dati fornite (Densità CSR/Sinteticità) se rilevanti per la profondità del report.
    - NON aggiungere conoscenza esterna, interpretazioni o "riempitive".
    - NON lodare l'azienda. Attieniti ai fatti sulla disponibilità dei dati.
    
    DATI INPUT:
    {data_json}
    
    METRICHE DI QUALITÀ:
    - Densità CSR: {csr_density} (Rapporto Segnale/Rumore)
    - Sinteticità: {conciseness} (Efficienza Output Strutturato)
    """
    
    # Convert report to JSON for the prompt
    data_json = report.model_dump_json(exclude={'initial_tokens', 'final_tokens', 'recap'})
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=RECAP_PROMPT.format(
            data_json=data_json,
            csr_density=report.csr_density,
            conciseness=report.conciseness_proxy
        )
    )
    
    if response.text:
        return response.text.strip()
    return "Nessuna sintesi generata."
