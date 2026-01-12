"""
EcoScan: Modelli di Mining di Sostenibilità Potenziato dall'IA
Strutture dati per l'estrazione degli Indicatori Chiave di Prestazione (KPI) ESG.
"""

from typing import Optional, Union
from pydantic import BaseModel, Field


class ESGKPI(BaseModel):
    """Modello generico per un singolo Indicatore Chiave di Prestazione ESG."""
    name: str = Field(description="Nome del KPI (es. 'Intensità GHG')")
    value: Union[str, float] = Field(description="Valore estratto (SOLO NUMERO come stringa, es. '550', '22.5'). NO testo/unità qui.")
    unit: str = Field(description="Unità di misura (es. 'tCO2e', '%', 'TRIR').")
    year: str = Field(..., description="Anno fiscale per questo punto dati")
    trend: Optional[str] = Field(None, description="Trend esplicitamente dichiarato (es. '-5%'). NON CALCOLARE.")
    standard_alignment: Optional[str] = Field(None, description="Standard utilizzato (es. 'GRI 305', 'SASB').")


class EnvironmentalPillar(BaseModel):
    """Metriche Pilastro E."""
    ghg_intensity: Optional[ESGKPI] = Field(None, description="E1. INTENSITÀ GHG (Scope 1+2+3)")
    renewable_energy: Optional[ESGKPI] = Field(None, description="E2. QUOTA ENERGIA RINNOVABILE")


class SocialPillar(BaseModel):
    """Metriche Pilastro S."""
    trir: Optional[ESGKPI] = Field(None, description="S1. TRIR (Tasso Totale Incidenti Registrabili)")
    women_in_leadership: Optional[ESGKPI] = Field(None, description="S2. DONNE IN LEADERSHIP (%)")


class GovernancePillar(BaseModel):
    """Metriche Pilastro G."""
    supplier_esg_score: Optional[ESGKPI] = Field(None, description="G1. PUNTEGGIO PERFORMANCE ESG FORNITORI")
    traceability: Optional[ESGKPI] = Field(None, description="G2. TRACCIABILITÀ SUPPLY CHAIN")


class ESGReport(BaseModel):
    """Dati ESG Completi estratti da un report."""
    company_name: str = Field(default="Sconosciuto", description="Nome dell'azienda")
    fiscal_year: str = Field(default="Sconosciuto", description="Anno fiscale del report")
    environment: EnvironmentalPillar = Field(default_factory=EnvironmentalPillar)
    social: SocialPillar = Field(default_factory=SocialPillar)
    governance: GovernancePillar = Field(default_factory=GovernancePillar)
    extraction_confidence: float = Field(default=0.0, description="Punteggio di confidenza dell'estrazione (0-1)")
    
    # Analisi del Sentiment (Nuovo Feature!)
    sentiment_score: float = Field(default=0.0, description="Compound score VADER (-1.0 a 1.0)")
    sentiment_label: str = Field(default="Neutrale", description="Etichetta sentiment (Positivo, Neutrale, Negativo)")

    
    # Metriche Metodologiche (Qualità Data Mining)
    csr_density: float = Field(default=0.0, description="Proxy per densità informativa (Frasi CSR / Frasi Totali)")
    conciseness_proxy: float = Field(default=0.0, description="Proxy per sinteticità testo (KPI estratti / Token Totali)")
    
    # Contabilità Token (Consapevolezza Costo/Efficienza)
    initial_tokens: int = Field(default=0, description="Token grezzi prima della pulizia")
    final_tokens: int = Field(default=0, description="Token inviati all'LLM")
    token_reduction_pct: float = Field(default=0.0, description="Guadagno efficienza (%)")
    
    # Riepilogo IA
    recap: Optional[str] = Field(None, description="Breve riepilogo esecutivo generato dall'IA della copertura dati")
