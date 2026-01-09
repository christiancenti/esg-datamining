# EcoScan: AI-Powered Sustainability Mining

This project demonstrates **EcoScan**, an **AI-powered ESG (Environmental, Social, Governance) Datamining Tool**.
It uses **Google Gemini 3.0 Flash Preview** (via native `google-genai` SDK) to extract key sustainability metrics from corporate reports (PDF/TXT) and visualizes them in an interactive dashboard.

## Overview

The application analyzes uploaded documents to extract 6 standardized KPIs:
*   **Greenhouse Gas Intensity (E1)**
*   **Renewable Energy Share (E2)**
*   **Total Recordable Incident Rate (S1)**
*   **Women in Leadership % (S2)**
*   **Supplier ESG Score (G1)**
*   **Supply Chain Traceability (G2)**

## Getting Started

### Prerequisites
*   Python 3.10+
*   Anaconda or Virtual Environment configured
*   Google Cloud API Key (for Gemini 3.0 Flash Preview)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Three-Pillars-Analytics/ecoscan.git
    cd ecoscan
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Using venv
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment:**
    **Option A: Local Development (.env)**
    Copy `.env.example` to `.env`:
    ```bash
    cp .env.example .env
    ```
    Edit `.env`:
    ```
    GOOGLE_API_KEY=your_api_key_here
    ```

    **Option B: Streamlit Cloud (Secrets)**
    Add your key to the Streamlit Secrets management or create `.streamlit/secrets.toml`:
    ```toml
    GOOGLE_API_KEY = "your_api_key_here"
    ```

### Usage

Run the dashboard application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

1. **Upload**: Drag & drop a PDF/TXT report.
2. **Demo Mode**: Click **"Use Example Report (Ferrero)"** in the sidebar to test the pipeline immediately with pre-loaded data.

### Example Output
- **Extracted KPIs**: 6/6 (100% Completeness)
- **CSR Density**: 0.42 (High Signal)
- **Conciseness Proxy**: 0.31
- **Token Reduction**: -66% (Cost Efficiency)
    
## 🔬 Methodology & Data Quality (CSRD aligned)

This project acts as a robust **Data Mining Pipeline** rather than a simple wrapper, aligned with academic literature on intellectual capital disclosure (e.g., Ricceri et al.).

### 1. Data Processing Pipeline
1.  **Ingestion**: High-fidelity PDF text extraction via `pdfplumber`.
2.  **Preprocessing & Noise Removal**:
    *   Removes headers, footers, page numbers, and artifacts.
    *   **Semantic Merging**: Reconstructs paragraphs to preserve context.
3.  **CSR Relevance Logic**: 
    *   A heuristic filter retains only ESG-relevant content.
    *   Calculates **CSR Density** (Information vs Noise ratio).
4.  **Token Accounting**: Tracks token usage pre/post cleaning (`tokens_raw` vs `tokens_clean`) to demonstrate cost-efficiency.

### 2. Structured Extraction & Logic
- **Deterministic Calculation**: The AI agent identifies raw numbers but uses a **deterministic Python tool** (`calculate_kpi`) to compute derived metrics (e.g., GHG Intensity), avoiding LLM math errors.
- **Single-Call Inference**: Leverages Gemini’s large context window to process the full cleaned document in a single inference call, avoiding fragmentation and context loss.
- **Strict Grounding**: The model extracts only explicit data; missing values returned as `null`.
- **Schema Validation**: Output enforced against a strict Pydantic `ESGReport` schema.

### 3. Quality Metrics (Proxies)
- **CSR Density**: Measures informative vs symbolic disclosure (Relevant Sentences / Total Sentences).
- **Conciseness Proxy**: Measures structured signal efficacy (Extracted KPI Tokens / Analyzed Tokens).

---

## ⚠️ Limitations

1.  **Not a Rating Tool**: This tool does not provide ESG ratings or scores, but focuses on data extraction and reporting quality analysis.
2.  **Proxy Metrics**: The quality metrics are operational proxies, not official standards.
3.  **Unimodal**: Text-only analysis; ignores charts/tables in images.
4.  **Dependency**: Output quality depends entirely on the source report's clarity.
5.  **Generative Probabilities**: Despite constraints, LLM interpretation retains a non-zero error margin.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments
 See `ACKNOWLEDGMENTS` for credits, references, and team contributions (Three Pillars Analytics).
