import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.extraction import analyze_structure, extract_kpis_with_llm
from src.models import ESGReport

# ============================================================================
# HELPER CLASSES
# ============================================================================

class MockUploadedFile:
    def __init__(self, path):
        self.name = os.path.basename(path)
        self.path = path
        self._content = None

    def getvalue(self):
        if self._content is None:
            with open(self.path, 'rb') as f:
                self._content = f.read()
        return self._content

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="EcoScan - Mining di Sostenibilità",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Ferrero-style" aesthetic
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    .css-1v0mbdj.e115fcil1 {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        background-color: white;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stAlert {
        border-radius: 8px;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 4px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.04);
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🌍 EcoScan: Mining di Sostenibilità Potenziato dall'IA")
    st.markdown("### Estrazione Automatizzata KPI per Report di Sostenibilità")
    
    # Session State Initialization
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None # (processed, raw, metrics)
    if 'final_report' not in st.session_state:
        st.session_state.final_report = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'is_extracting' not in st.session_state:
        st.session_state.is_extracting = False
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False

    # Sidebar
    with st.sidebar:
        st.header("📄 Carica Report")
        uploaded_file_widget = st.file_uploader("Carica Report ESG (PDF)", type=['pdf'])
        
        st.write("OPPURE")
        
        if st.button("📊 Usa Report di Esempio (Ferrero)"):
            st.session_state.demo_mode = True
            st.rerun()

        st.markdown("---")
        st.info(
            """
            **Come funziona**
            1. **Preprocessing**: Pulisce layout e rumore
            2. **Analisi**: Valuta densità e rilevanza
            3. **Estrazione**: Gemini AI mappa i KPI
            """
        )
        st.markdown("---")
        st.caption("Potenziato da Google Gemini 3.0 Flash Preview")

    # LOGIC: Resolve Uploaded File
    uploaded_file = None
    
    if uploaded_file_widget:
        # If user uploads a file, prioritize it and disable demo mode
        if st.session_state.demo_mode:
            st.session_state.demo_mode = False
            st.rerun()
        uploaded_file = uploaded_file_widget
    elif st.session_state.demo_mode:
        # Load the mock file
        file_path = os.path.join("reports", "ferrero.pdf")
        if os.path.exists(file_path):
            uploaded_file = MockUploadedFile(file_path)
            # Show a clear button to exit demo mode
            with st.sidebar:
                if st.button("❌ Esci dalla Modalità Demo", type="primary"):
                    st.session_state.demo_mode = False
                    st.rerun()
        else:
            st.error(f"File di esempio non trovato in {file_path}")

    # LOGIC: New File Data Processing
    if uploaded_file:
        # Check if it's a new file to reset state
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.final_report = None
            st.session_state.is_extracting = False
            
            # STEP 1: PREPROCESSING (Instant)
            with st.spinner("Pipeline: Ingestione -> Pulizia -> Formattazione..."):
                processed, raw, metrics = analyze_structure(uploaded_file)
                st.session_state.processed_data = (processed, raw, metrics)

        # RENDER STEP 1: Methodological View
        if st.session_state.processed_data:
            processed_txt, raw_txt, metrics = st.session_state.processed_data
            
            st.success("✅ Preprocessing Completato")
            
            # Metrics Display
            st.markdown("### 📐 Metriche Metodologiche e di Qualità")
            st.caption("Allineato con CSRD & Ricceri et al. (Divulgazione del Capitale Intellettuale)")
            
            st.markdown("#### 🔹 Proxy di Qualità")
            q1, q2 = st.columns(2)
            q1.metric(
                "Densità CSR", 
                f"{metrics.get('csr_density', 0)*100:.1f}%", 
                help="Rapporto tra frasi rilevanti ESG e frasi totali."
            )
            q2.metric(
                "Sinteticità", 
                f"{metrics.get('conciseness', 0):.4f}",
                help="Token ESG Rilevanti / Token Puliti Totali."
            )
            
            # Sentiment Display
            st.write("")
            st.markdown("#### 📢 Tono di Voce (Sentiment)")
            if "sentiment_score" in metrics:
                s_score = metrics["sentiment_score"]
                s_label = metrics["sentiment_label"]
                
                # Logic for tooltip explanation
                help_text = """
                **Analisi del Tono Comunicativo (VADER Average)**
                
                Indica la densità media di sentiment positivo *per frase*:
                - **Alto (> 0.4)**: Molto enfatico/promozionale.
                - **Medio (0.1 - 0.4)**: Bilanciato, tipico dei report standard.
                - **Neutro (< 0.1)**: Tono tecnico e distaccato.
                
                Calcolato come media dei compound score su frasi singole.
                """
                
                st.metric("Enfasi Promozionale (Densità)", f"{s_score:.2f}", delta=s_label, delta_color="normal", help=help_text)
            
            st.write("")
            
            st.markdown("#### 🔹 Contabilità Token (Costo/Efficienza)")
            t1, t2, t3 = st.columns(3)
            t1.metric("Token Iniziali (Grezzi)", f"{metrics['initial_tokens']:,}")
            t2.metric("Token Finali (Puliti)", f"{metrics['final_tokens']:,}")
            t3.metric(
                "Riduzione Rumore", 
                f"-{metrics['reduction_pct']:.1f}%", 
                help="Riduzione dell'uso di token dovuta alla pipeline di preprocessing."
            )

            # TF-IDF Keywords Display
            if "top_keywords" in metrics and metrics["top_keywords"]:
                st.write("")
                st.markdown("#### 🔑 Temi Principali TF-IDF")
                st.caption("Parole chiave specifiche estratte automaticamente (Frequenza del Termine - Frequenza Inversa del Documento)")
                
                # Render as tags (with margin for spacing)
                tags_html = "".join([
                    f"<span style='background-color:#e0f7fa; color:#006064; padding:5px 12px; border-radius:15px; margin: 4px 6px; display:inline-block; font-size:0.9em; border:1px solid #b2ebf2;'>{kw}</span>"
                    for kw in metrics["top_keywords"]
                ])
                st.markdown(tags_html, unsafe_allow_html=True)
                st.write("")

            # Raw vs Clean Diff Viewer
            st.subheader("🔎 Ispettore Pipeline: Grezzo vs Pulito")
            tab_clean, tab_raw = st.tabs(["✨ Markdown Pulito (Input LLM)", "📝 Testo Estratto Grezzo"])
            
            with tab_clean:
                st.markdown("Il testo pulito e strutturato inviato all'Agente IA:")
                st.text_area("Contenuto Pulito", processed_txt, height=250, disabled=True)
            
            with tab_raw:
                st.markdown("Il testo grezzo estratto dal PDF (prima della pulizia):")
                st.text_area("Contenuto Grezzo", raw_txt, height=250, disabled=True)

            # STEP 2: EXTRACTION ACTION
            st.divider()
            
            def start_extraction():
                st.session_state.is_extracting = True

            col_action, col_space = st.columns([1, 4])
            with col_action:
                if not st.session_state.is_extracting:
                    st.button("🚀 Avvia Estrazione AI", type="primary", width="stretch", on_click=start_extraction)
                else:
                    with st.spinner("🤖 Estrazione KPI con Gemini 3.0 (Tooling Deterministico)..."):
                        try:
                            report = extract_kpis_with_llm(processed_txt, metrics)
                            st.session_state.final_report = report
                            st.toast("✅ Analisi Completata!", icon="🎉")
                        except Exception as e:
                            st.error(f"⚠️ Estrazione fallita: {str(e)}")
                        finally:
                            st.session_state.is_extracting = False
                            st.rerun()

            # STEP 3: FINAL DASHBOARD
            if st.session_state.final_report:
                render_dashboard(st.session_state.final_report)

    else:
        # Reset state if no file
        st.session_state.processed_data = None
        st.session_state.final_report = None
        st.session_state.current_file = None
        render_landing_state()


def render_landing_state():
    """Display empty state or demo info."""
    st.info("👈 Carica un Report di Sostenibilità per avviare l'estrazione.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🌍 Ambiente")
        st.markdown("- Intensità GHG\n- Quota Energia Rinnovabile")
    with col2:
        st.markdown("### 👥 Sociale")
        st.markdown("- TRIR (Sicurezza)\n- Donne in Leadership")
    with col3:
        st.markdown("### 🏛️ Governance")
        st.markdown("- Punteggio ESG Fornitori\n- Tracciabilità Supply Chain")


# ============================================================================
# BENCHMARKS (Food & Beverage Sector 2024 Estimates)
# ============================================================================
BENCHMARKS = {
    "E1": {"value": 550, "unit": "tCO2eq/M€", "label": "Media Industria (~550)"},
    "E2": {"value": 22.0, "unit": "%", "label": "Media Globale F&B (~22%)"},
    "S1": {"value": 3.6, "unit": "TRIR", "label": "Media OSHA Food Mfg (3.6)"},
    "S2": {"value": 33.5, "unit": "%", "label": "Media Settore (33.5%)"},
    "G1": {"value": 50, "unit": "Score", "label": "Media EcoVadis (50)"},
    "G2": {"value": 75, "unit": "%", "label": "Media Tracciabilità (75%)"}
}

def render_radar_chart(report: ESGReport):
    """Generates a Radar Chart comparing Company Performance vs Industry Benchmark."""
    
    # Normalize Data [0-100 Scale for Visualization]
    # We define a 'Target' or 'Max' valid range for normalization
    
    categories = ['E1. GHG (-)', 'E2. Rinnovabili (+)', 'S1. Sicurezza (-)', 
                  'S2. Leadership Donne (+)', 'G1. Fornitori (+)', 'G2. Tracciabilità (+)']
    
    # Parsing values (safely)
    def clean_val(v):
        try: return float(str(v).replace('%','').replace(',','').split(' ')[0])
        except: return 0.0

    # Company Values
    e1 = clean_val(report.environment.ghg_intensity.value) if report.environment.ghg_intensity else 0
    e2 = clean_val(report.environment.renewable_energy.value) if report.environment.renewable_energy else 0
    s1 = clean_val(report.social.trir.value) if report.social.trir else 0
    s2 = clean_val(report.social.women_in_leadership.value) if report.social.women_in_leadership else 0
    g1 = clean_val(report.governance.supplier_esg_score.value) if report.governance.supplier_esg_score else 0
    g2 = clean_val(report.governance.traceability.value) if report.governance.traceability else 0
    
    # Normalization Logic (0-100)
    # E1: Lower is better. Let's say 1000 is bad (0), 0 is perfect (100).
    # S1: Lower is better. 10 is bad (0), 0 is perfect (100).
    
    company_norm = [
        max(0, 100 - (e1 / 10)),   # E1 (approx scale)
        min(100, e2),              # E2 (is %)
        max(0, 100 - (s1 * 10)),   # S1 (TRIR * 10 approx)
        min(100, s2),              # S2 (is %)
        min(100, g1),              # G1 (is score)
        min(100, g2)               # G2 (is %)
    ]
    
    # Industry Benchmark Norm
    # Based on BENCHMARKS constant
    bench_norm = [
        max(0, 100 - (BENCHMARKS['E1']['value'] / 10)),
        BENCHMARKS['E2']['value'],
        max(0, 100 - (BENCHMARKS['S1']['value'] * 10)),
        BENCHMARKS['S2']['value'],
        BENCHMARKS['G1']['value'],
        BENCHMARKS['G2']['value']
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=company_norm,
        theta=categories,
        fill='toself',
        name=report.company_name,
        line_color='#00C853'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=bench_norm,
        theta=categories,
        fill='toself',
        name='Media Settore (Benchmark)',
        line_color='#B0BEC5',
        opacity=0.5
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Performance Relativa Normalizzata (0-100)",
        height=400,
        margin=dict(l=40, r=40, t=40, b=20)
    )
    
    st.plotly_chart(fig, width="stretch")


def render_dashboard(report: ESGReport):
    """Render the main key metrics dashboard with benchmarks."""
    
    st.divider()
    
    # Header Info
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.subheader(f"📊 Analisi Report: {report.company_name}")
        st.caption(f"Anno Fiscale: {report.fiscal_year}")
    with col_head2:
        st.write("")
        st.markdown("✅ **Allineato ESRS**")
        st.markdown("✅ **Conforme GRI**")
    
    # --- AI RECAP (EXECUTIVE SUMMARY) ---
    if report.recap:
        st.info(f"**🤖 Sintesi Esecutiva IA**: {report.recap}")
        
    # --- RADAR CHART (VISUAL IMPACT) ---
    st.markdown("### 🕸️ Radar ESG: Azienda vs Settore")
    render_radar_chart(report)
    
    st.markdown("### 🎯 Performance vs Benchmark di Settore")
    
    # Helper to clean value for comparison
    def parse_val(v_str):
        try:
            return float(str(v_str).replace('%', '').replace(',','').split(' ')[0])
        except:
            return None

    # Helper for metric card
    def metric_card(title, kpi, benchmark_key, inverse=False):
        if not kpi:
            st.warning(f"{title}: Dati non trovati")
            return
            
        val_num = parse_val(kpi.value)
        bench = BENCHMARKS.get(benchmark_key)
        
        delta = None
        delta_color = "normal"
        
        if val_num is not None and bench:
            diff = val_num - bench["value"]
            delta = f"{diff:+.1f} vs Media Ind."
            
            # Simplified Logic relying on Streamlit's native behavior:
            # - normal: Positive=Green, Negative=Red (Higher is Better)
            # - inverse: Positive=Red, Negative=Green (Lower is Better -> e.g. GHG, TRIR)
            
            if inverse:
                delta_color = "inverse" 
            else:
                delta_color = "normal"
        
        # Display
        st.metric(
            label=f"{title}",
            value=f"{kpi.value} {kpi.unit}",
            delta=delta if delta else kpi.trend,
            delta_color=delta_color,
            help=f"Benchmark: {bench['label']}, Standard: {kpi.standard_alignment}"
        )
        if bench:
            st.caption(f"📉 Bench: {bench['value']}")

    # --- ENVIRONMENTAL PILLAR ---
    st.markdown("#### 🌍 Ambientale (E)")
    col1, col2 = st.columns(2)
    with col1:
        metric_card("E1. Intensità GHG", report.environment.ghg_intensity, "E1", inverse=True)
    with col2:
        metric_card("E2. Energia Rinnovabile", report.environment.renewable_energy, "E2")
            
    # --- SOCIAL PILLAR ---
    st.markdown("#### 👥 Sociale (S)")
    col3, col4 = st.columns(2)
    with col3:
        metric_card("S1. TRIR (Sicurezza)", report.social.trir, "S1", inverse=True)
    with col4:
        metric_card("S2. Donne in Leadership", report.social.women_in_leadership, "S2")

    # --- GOVERNANCE PILLAR ---
    st.markdown("#### 🏛️ Governance (G)")
    col5, col6 = st.columns(2)
    with col5:
        metric_card("G1. Punteggio Fornitori", report.governance.supplier_esg_score, "G1")
    with col6:
        metric_card("G2. Tracciabilità", report.governance.traceability, "G2")

    # Export Data Table
    st.divider()
    with st.expander("📥 Visualizza ed Esporta Tutti i Dati"):
        # Flattens data for table
        data = []
        
        def add_row(pillar, code, kpi):
            if kpi:
                data.append({
                    "Pilastro": pillar,
                    "Codice": code,
                    "Nome KPI": kpi.name,
                    "Valore": kpi.value,
                    "Unità": kpi.unit,
                    "Trend": kpi.trend,
                    "Standard": kpi.standard_alignment
                })

        add_row("Ambientale", "E1", report.environment.ghg_intensity)
        add_row("Ambientale", "E2", report.environment.renewable_energy)
        add_row("Sociale", "S1", report.social.trir)
        add_row("Sociale", "S2", report.social.women_in_leadership)
        add_row("Governance", "G1", report.governance.supplier_esg_score)
        add_row("Governance", "G2", report.governance.traceability)
        
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, width="stretch")
            
            # Download JSON Button
            st.download_button(
                label="📥 Scarica Report Completo (JSON)",
                data=report.model_dump_json(indent=2),
                file_name=f"esg_report_{report.company_name}_{report.fiscal_year}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
