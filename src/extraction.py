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
You are an expert ESG Analyst for a top-tier consulting firm.
Your task is to extract structured ESG data from corporate sustainability reports.

You will receive the text of a report (already pre-processed and cleaned).
You must extract 6 specific Key Performance Indicators (KPIs) and map them to the provided JSON schema.

## KPI DEFINITIONS:
1. **GHG Intensity (E1)**: (Environmental) Scope 1+2 emissions normalized by revenue (e.g., tCO2e/€M).
2. **Renewable Energy Share (E2)**: (Environmental) Percentage of electricity from renewable sources.
3. **TRIR (S1)**: (Social) Total Recordable Incident Rate (Safety metric).
4. **Women in Leadership (S2)**: (Social) Percentage of women in management/executive roles.
5. **Supplier ESG Score (G1)**: (Governance) Average score or assessment coverage of suppliers (e.g., EcoVadis, Sedex).
6. **Supply Chain Traceability (G2)**: (Governance) Percentage of raw materials traceable to the source.

## CRITICAL RULES:
- **FORMATTING**:
    - `value`: MUST be a clean number/percentage string (e.g., "550", "22.5", "3.6"). DO NOT include text here.
    - `unit`: Extract the unit separately (e.g., "tCO2e/€M", "%", "TRIR", "Score").
    - `trend`: EXTRACT ONLY if explicitly stated in text (e.g. "decreased by 5%"). DO NOT CALCULATE IT.
- **CONCISENESS**: Text fields (standard) must be SHORT.
- **Fiscal Year**: Extract the fiscal year (e.g., "FY 2023", "2023").
- **Company Name**: Extract the exact company name.
- **Trends**: DO NOT CALCULATE TRENDS between years. Only extract explicit statements.
- If a metric is NOT found, leave it as null (do not hallucinate).
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
        raise ValueError("GOOGLE_API_KEY not found in st.secrets or environment variables.")
    
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
    
    log(f"🚀 Preprocessing: {filename}")
    
    
    if filename.endswith('.pdf'):
        processed_text, raw_text, metrics = process_pdf_pipeline(file_bytes)
        
        # Calculate TF-IDF Top Keywords (using processed text for quality)
        log("   🧮 Calculating TF-IDF Top Themes...")
        top_kwd = extract_top_keywords(processed_text, top_n=8)
        metrics["top_keywords"] = top_kwd
        
        log(f"   📊 Token Efficiency: ▼ {metrics['reduction_pct']}%")
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

        # Define Deterministic Tool (with logging wrapper)
        def calculate_kpi(numerator: float, denominator: float) -> float:
            """Calculates a KPI intensity or ratio (numerator / denominator)."""
            log(f"      🧮 Tool Triggered: {numerator} / {denominator}")
            if denominator == 0: return 0.0
            res = round(numerator / denominator, 4)
            log(f"      ↳ Result: {res}")
            return res

        # Update Prompt
        TOOL_PROMPT_ADDITION = """
## CALCULATION RULES:
- If you find raw numbers (e.g. Total Emissions = 1000, Revenue = 50), DO NOT calculate the intensity yourself.
- USE the `calculate_kpi` tool to compute the result deterministically.
- E.g. call calculate_kpi(1000, 50) -> 20.0

## STRICT GROUNDING RULE:
- EXTRACT ONLY DATA EXPLICITLY PRESENT IN THE TEXT.
- DO NOT INFER OR GUESS MISSING VALUES.
- If a value is missing, return null.
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
            f"{ESG_EXTRACTION_PROMPT_SYSTEM}\n{TOOL_PROMPT_ADDITION}\n\nHere is the report content:\n\n{processed_text}"
        )

        # Handle Function Calls (Manual Loop for control/logging)
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

        # --- GENERATE RECAP (SAFE DATA-ONLY SUMMARY) ---
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


# Backward compatibility wrapper
def process_esg_report(uploaded_file, log_callback: Optional[Callable[[str], None]] = None) -> ESGReport:
    processed, _, metrics = analyze_structure(uploaded_file, log_callback)
    if not processed: return ESGReport()
    return extract_kpis_with_llm(processed, metrics, log_callback)


# ============================================================================
# RECAP GENERATOR
# ============================================================================

def generate_data_recap(client: genai.Client, report: ESGReport) -> str:
    """
    Generate a strictly data-grounded executive summary using standard text generation.
    """
    
    RECAP_PROMPT = """You are an ESG Data Analyst.
    Your task is to write a SHORT, NEUTRAL executive summary (max 4-5 sentences) based ONLY on the provided structured data.
    
    GUIDELINES:
    - Summarize which ESG areas (Environment, Social, Governance) have data coverage.
    - Explicitly mention missing or incomplete metrics.
    - Mention positive/negative trends if they are in the data.
    - Reference the Data Quality Metrics provided (CSR Density/Conciseness) if relevant to the report's depth.
    - DO NOT add external knowledge, interpretations, or "fluff".
    - DO NOT praise the company. Stick to the facts of data availability.
    
    INPUT DATA:
    {data_json}
    
    QUALITY METRICS:
    - CSR Density: {csr_density} (Signal/Noise ratio)
    - Conciseness: {conciseness} (Structured Output efficiency)
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
    return "No summary generated."
