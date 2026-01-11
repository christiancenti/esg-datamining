import os
import streamlit as st
import pandas as pd
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
    page_title="EcoScan - Sustainability Mining",
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
    st.title("🌍 EcoScan: AI-Powered Sustainability Mining")
    st.markdown("### Automated KPI Extraction for Sustainability Reports")
    
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
        st.header("📄 Upload Report")
        uploaded_file_widget = st.file_uploader("Upload ESG Report (PDF)", type=['pdf'])
        
        st.write("OR")
        
        if st.button("📊 Use Example Report (Ferrero)"):
            st.session_state.demo_mode = True
            st.rerun()

        st.markdown("---")
        st.info(
            """
            **How it works**
            1. **Preprocessing**: Cleans layout & noise
            2. **Analysis**: Evaluates density & relevance
            3. **Extraction**: Gemini AI maps to KPIs
            """
        )
        st.markdown("---")
        st.caption("Powered by Google Gemini 3.0 Flash Preview")

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
                if st.button("❌ Exit Demo Mode", type="primary"):
                    st.session_state.demo_mode = False
                    st.rerun()
        else:
            st.error(f"Example file not found at {file_path}")

    # LOGIC: New File Data Processing
    if uploaded_file:
        # Check if it's a new file to reset state
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.final_report = None
            st.session_state.is_extracting = False
            
            # STEP 1: PREPROCESSING (Instant)
            with st.spinner("Pipeline: Ingestion -> Cleaning -> Formatting..."):
                processed, raw, metrics = analyze_structure(uploaded_file)
                st.session_state.processed_data = (processed, raw, metrics)

        # RENDER STEP 1: Methodological View
        if st.session_state.processed_data:
            processed_txt, raw_txt, metrics = st.session_state.processed_data
            
            st.success("✅ Preprocessing Complete")
            
            # Metrics Display
            st.markdown("### 📐 Methodological & Quality Metrics")
            st.caption("Aligned with CSRD & Ricceri et al. (Intellectual Capital Disclosure)")
            
            st.markdown("#### 🔹 Quality Proxies")
            q1, q2 = st.columns(2)
            q1.metric(
                "CSR Density", 
                f"{metrics.get('csr_density', 0)*100:.1f}%", 
                help="Ratio of ESG-relevant sentences vs Total sentences."
            )
            q2.metric(
                "Conciseness", 
                f"{metrics.get('conciseness', 0):.4f}",
                help="Relevant ESG Tokens / Total Cleaned Tokens."
            )
            
            st.write("") # Spacer
            
            st.markdown("#### 🔹 Token Accounting (Cost/Efficiency)")
            t1, t2, t3 = st.columns(3)
            t1.metric("Initial Tokens (Raw)", f"{metrics['initial_tokens']:,}")
            t2.metric("Final Tokens (Clean)", f"{metrics['final_tokens']:,}")
            t3.metric(
                "Noise Reduction", 
                f"-{metrics['reduction_pct']:.1f}%", 
                help="Reduction in token usage due to preprocessing pipeline."
            )

            # TF-IDF Keywords Display
            if "top_keywords" in metrics and metrics["top_keywords"]:
                st.write("")
                st.markdown("#### 🔑 TF-IDF Top Themes")
                st.caption("Auto-extracted specific keywords (Term Frequency - Inverse Document Frequency)")
                
                # Render as tags (with margin for spacing)
                tags_html = "".join([
                    f"<span style='background-color:#e0f7fa; color:#006064; padding:5px 12px; border-radius:15px; margin: 4px 6px; display:inline-block; font-size:0.9em; border:1px solid #b2ebf2;'>{kw}</span>"
                    for kw in metrics["top_keywords"]
                ])
                st.markdown(tags_html, unsafe_allow_html=True)
                st.write("")

            # Raw vs Clean Diff Viewer
            st.subheader("🔎 Pipeline Inspector: Raw vs Clean")
            tab_clean, tab_raw = st.tabs(["✨ Cleaned Markdown (LLM Input)", "📝 Raw Extracted Text"])
            
            with tab_clean:
                st.markdown("The cleaned, structured text sent to the AI Agent:")
                st.text_area("Cleaned Content", processed_txt, height=250, disabled=True)
            
            with tab_raw:
                st.markdown("The raw text extracted from the PDF (before cleaning):")
                st.text_area("Raw Content", raw_txt, height=250, disabled=True)

            # STEP 2: EXTRACTION ACTION
            st.divider()
            
            def start_extraction():
                st.session_state.is_extracting = True

            col_action, col_space = st.columns([1, 4])
            with col_action:
                if not st.session_state.is_extracting:
                    st.button("🚀 Avvia Estrazione AI", type="primary", width="stretch", on_click=start_extraction)
                else:
                    with st.spinner("🤖 Extracting KPIs with Gemini 3.0 (Deterministic Tooling)..."):
                        try:
                            report = extract_kpis_with_llm(processed_txt, metrics)
                            st.session_state.final_report = report
                            st.toast("✅ Analysis Complete!", icon="🎉")
                        except Exception as e:
                            st.error(f"⚠️ Extraction failed: {str(e)}")
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
    st.info("👈 Upload a Sustainability Report to start extraction.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🌍 Environment")
        st.markdown("- GHG Intensity\n- Renewable Energy Share")
    with col2:
        st.markdown("### 👥 Social")
        st.markdown("- TRIR (Safety)\n- Women in Leadership")
    with col3:
        st.markdown("### 🏛️ Governance")
        st.markdown("- Supplier ESG Score\n- Supply Chain Traceability")


# ============================================================================
# BENCHMARKS (Food & Beverage Sector 2024 Estimates)
# ============================================================================
BENCHMARKS = {
    "E1": {"value": 550, "unit": "tCO2eq/M€", "label": "Industry Avg (~550)"},
    "E2": {"value": 22.0, "unit": "%", "label": "Global F&B Avg (~22%)"},
    "S1": {"value": 3.6, "unit": "TRIR", "label": "OSHA Food Mfg Avg (3.6)"},
    "S2": {"value": 33.5, "unit": "%", "label": "Sector Avg (33.5%)"},
    "G1": {"value": 50, "unit": "Score", "label": "EcoVadis Avg (50)"},
    "G2": {"value": 75, "unit": "%", "label": "Avg Traceability (75%)"}
}

def render_dashboard(report: ESGReport):
    """Render the main key metrics dashboard with benchmarks."""
    
    st.divider()
    
    # Header Info
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.subheader(f"📊 Report Analysis: {report.company_name}")
        st.caption(f"Fiscal Year: {report.fiscal_year}")
    with col_head2:
        st.write("") # Spacer
        st.markdown("✅ **ESRS Aligned**")
        st.markdown("✅ **GRI Compliant**")
    
    # --- AI RECAP (EXECUTIVE SUMMARY) ---
    if report.recap:
        st.info(f"**🤖 AI Executive Summary**: {report.recap}")
    
    st.markdown("### 🎯 Performance vs Industry Benchmarks")
    
    # Helper to clean value for comparison
    def parse_val(v_str):
        try:
            return float(str(v_str).replace('%', '').replace(',','').split(' ')[0])
        except:
            return None

    # Helper for metric card
    def metric_card(title, kpi, benchmark_key, inverse=False):
        if not kpi:
            st.warning(f"{title}: Data not found")
            return
            
        val_num = parse_val(kpi.value)
        bench = BENCHMARKS.get(benchmark_key)
        
        delta = None
        delta_color = "normal"
        
        if val_num is not None and bench:
            diff = val_num - bench["value"]
            delta = f"{diff:+.1f} vs Ind. Avg"
            
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
    st.markdown("#### 🌍 Environmental (E)")
    col1, col2 = st.columns(2)
    with col1:
        metric_card("E1. GHG Intensity", report.environment.ghg_intensity, "E1", inverse=True)
    with col2:
        metric_card("E2. Renewable Energy", report.environment.renewable_energy, "E2")
            
    # --- SOCIAL PILLAR ---
    st.markdown("#### 👥 Social (S)")
    col3, col4 = st.columns(2)
    with col3:
        metric_card("S1. TRIR (Safety)", report.social.trir, "S1", inverse=True)
    with col4:
        metric_card("S2. Women Leadership", report.social.women_in_leadership, "S2")

    # --- GOVERNANCE PILLAR ---
    st.markdown("#### 🏛️ Governance (G)")
    col5, col6 = st.columns(2)
    with col5:
        metric_card("G1. Supplier Score", report.governance.supplier_esg_score, "G1")
    with col6:
        metric_card("G2. Traceability", report.governance.traceability, "G2")

    # Export Data Table
    st.divider()
    with st.expander("📥 View & Export All Data"):
        # Flattens data for table
        data = []
        
        def add_row(pillar, code, kpi):
            if kpi:
                data.append({
                    "Pillar": pillar,
                    "Code": code,
                    "KPI Name": kpi.name,
                    "Value": kpi.value,
                    "Unit": kpi.unit,
                    "Trend": kpi.trend,
                    "Standard": kpi.standard_alignment
                })

        add_row("Environmental", "E1", report.environment.ghg_intensity)
        add_row("Environmental", "E2", report.environment.renewable_energy)
        add_row("Social", "S1", report.social.trir)
        add_row("Social", "S2", report.social.women_in_leadership)
        add_row("Governance", "G1", report.governance.supplier_esg_score)
        add_row("Governance", "G2", report.governance.traceability)
        
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, width="stretch")


if __name__ == "__main__":
    main()
