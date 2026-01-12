# EcoScan: Mining di Sostenibilità Potenziato dall'IA

Questo progetto dimostra **EcoScan**, uno **Strumento di Datamining ESG (Environmental, Social, Governance) potenziato dall'IA**.
Utilizza **Google Gemini 3.0 Flash Preview** (tramite SDK nativo `google-genai`) per estrarre metriche chiave di sostenibilità dai report aziendali (PDF) e visualizzarle in una dashboard interattiva.

## Panoramica

L'applicazione analizza i documenti caricati per estrarre 6 KPI standardizzati:
*   **Intensità di Gas Serra (E1)**
*   **Quota di Energia Rinnovabile (E2)**
*   **Tasso Totale di Incidenti Registrabili (S1)**
*   **Donne in Ruoli di Leadership % (S2)**
*   **Punteggio ESG dei Fornitori (G1)**
*   **Tracciabilità della Catena di Fornitura (G2)**

Inoltre, fornisce un'analisi qualitativa avanzata includendo:
- **Analisi del Tono Comunicativo** per misurare l'enfasi promozionale.
- **Grafico Radar Benchmark** per il posizionamento competitivo rispetto al settore.

## Per Iniziare

### Prerequisiti
*   Python 3.10+
*   Anaconda o Ambiente Virtuale configurato
*   Chiave API di Google Cloud (per Gemini 3.0 Flash Preview)

### Installazione

1.  **Clona il repository:**
    ```bash
    git clone https://github.com/Three-Pillars-Analytics/ecoscan.git
    cd ecoscan
    ```

2.  **Crea e attiva un ambiente virtuale:**
    ```bash
    # Usando venv
    python -m venv venv
    source venv/bin/activate  # Su Windows: venv\Scripts\activate
    ```

3.  **Installa le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configura l'Ambiente:**
    **Opzione A: Sviluppo Locale (.env)**
    Copia `.env.example` in `.env`:
    ```bash
    cp .env.example .env
    ```
    Modifica `.env`:
    ```
    GOOGLE_API_KEY=la_tua_api_key_qui
    ```

    **Opzione B: Streamlit Cloud (Secrets)**
    Aggiungi la tua chiave alla gestione dei Segreti di Streamlit o crea `.streamlit/secrets.toml`:
    ```toml
    GOOGLE_API_KEY = "la_tua_api_key_qui"
    ```

### Utilizzo

Esegui l'applicazione dashboard:
```bash
streamlit run app.py
```

L'app si aprirà nel tuo browser all'indirizzo `http://localhost:8501`.

1. **Carica**: Trascina e rilascia un report PDF.
2. **Modalità Demo**: Clicca su **"Usa Report di Esempio (Ferrero)"** nella barra laterale per testare immediatamente la pipeline con dati pre-caricati.

### Esempio di Output
- **KPI Estratti**: 6/6 (100% Completezza)
- **Densità CSR**: 0.42 (Segnale Alto)
- **Proxy di Sinteticità**: 0.31
- **Riduzione Token**: -66% (Efficienza dei Costi)
    
## 🔬 Metodologia e Qualità dei Dati (Allineato CSRD)

Questo progetto agisce come una robusta **Pipeline di Data Mining** piuttosto che un semplice wrapper, allineato con la letteratura accademica sulla divulgazione del capitale intellettuale (ad es. Ricceri et al.).

### 1. Pipeline di Elaborazione Dati
1.  **Ingestione**: Estrazione di testo ad alta fedeltà da PDF tramite `pdfplumber`.
2.  **Preprocessing Avanzato & Riduzione del Rumore**:
    *   **Pulizia del Layout**: Rimozione basata su Regex di elementi non semantici (intestazioni, piè di pagina, artefatti di impaginazione).
    *   **Rimozione Stopword (NLTK)**: Filtro di Elaborazione del Linguaggio Naturale (NLP) per eliminare stopword a bassa informazione (supporto multilingue), riducendo la dimensionalità dei token pur preservando la struttura della frase.
    *   **Fusione Semantica**: Ricostruzione di flussi di testo interrotti in paragrafi coerenti.
3.  **Filtraggio Argomenti basato su Dizionario**: 
    *   **Estrazione Basata su Regole**: Una fase di filtraggio supervisionato che conserva solo i paragrafi contenenti parole chiave ESG specifiche del dominio (Filtraggio per Rilevanza).
    *   **Calcolo Densità CSR**: Calcola il rapporto Segnale-Rumore (Contenuto Rilevante / Contenuto Totale) per quantificare la densità del report.
4.  **Estrazione Parole Chiave TF-IDF**: Utilizza **TI-IDF** (Term Frequency-Inverse Document Frequency) tramite `scikit-learn` per estrarre i "Temi Principali".
    *   **Perché TF-IDF?**: A differenza dei semplici conteggi di frequenza (Bag-of-Words), TF-IDF penalizza i termini generici che appaiono ovunque (es. "sostenibilità"), dando priorità a concetti specifici ad alta densità (es. "imballaggio", "diritti umani").
    *   **Filtro Aziendale**: Include una lista di esclusione personalizzata per il linguaggio standard (es. "gruppo", "continuato", "anno fiscale").
5.  **Contabilità Token**: Traccia l'utilizzo dei token pre/post pulizia (`tokens_raw` vs `tokens_clean`) per dimostrare l'efficienza dei costi e i tassi di compressione.

### 2. Estrazione Strutturata & Logica
- **Calcolo Deterministico**: L'agente IA identifica i numeri grezzi ma utilizza uno **strumento Python deterministico** (`calculate_kpi`) per calcolare le metriche derivate (es. Intensità GHG), evitando errori matematici dell'LLM.
- **Inferenza a Chiamata Singola**: Sfrutta l'ampia finestra di contesto di Gemini per elaborare l'intero documento pulito in una singola chiamata di inferenza, evitando frammentazione e perdita di contesto.
- **Strict Grounding**: Il modello estrae solo dati espliciti; i valori mancanti sono restituiti come `null`.
- **Convalida Schema**: Output forzato contro un rigoroso schema Pydantic `ESGReport`.

### 3. Metriche di Qualità (Proxy)
- **Densità CSR**: Misura la divulgazione informativa vs simbolica (Frasi Rilevanti / Frasi Totali).
- **Proxy di Sinteticità**: Misura l'efficacia del segnale strutturato (Token KPI Estratti / Token Analizzati).
- **Analisi Tono Comunicativo**: Utilizza **NLTK VADER** per calcolare una "Densità di Enfasi Promozionale" (media del sentiment per frase), distinguendo tra linguaggio tecnico (neutro) e marketing (enfatico/promozionale).

---

## ⚠️ Limitazioni

1.  **Non uno Strumento di Rating**: Questo strumento non fornisce rating o punteggi ESG, ma si concentra sull'estrazione dei dati e sull'analisi della qualità del reporting.
2.  **Metriche Proxy**: Le metriche di qualità sono proxy operativi, non standard ufficiali.
3.  **Unimodale**: Analisi solo testo; ignora grafici/tabelle nelle immagini.
    TODO: In una fase successiva, si possono estrarre le immagini dal pdf e farle analizzare a Gemini per estrarre ulteriori informazioni. (Gemini 3.0 Flash Preview è multimodale)
4.  **Dipendenza**: La qualità dell'output dipende interamente dalla chiarezza del report sorgente.
5.  **AI Generativa**: Nonostante i vincoli, l'interpretazione LLM mantiene un margine di errore non nullo.

## Licenza
Distribuito sotto la Licenza MIT. Vedi `LICENSE` per maggiori informazioni.
*Scelta per la sua natura permissiva e open-source. Il software è fornito "AS IS" (così com'è) senza garanzie; gli autori declinano ogni responsabilità sull'accuratezza dei dati finanziari/ESG estratti.*

## Riconoscimenti
 Vedi `ACKNOWLEDGMENTS` per crediti, riferimenti e contributi del team (Three Pillars Analytics).
