"""
EcoScan: AI-Powered Sustainability Mining Models
Data structures for ESG Key Performance Indicator (KPI) extraction.
"""

from typing import Optional, Union
from pydantic import BaseModel, Field


class ESGKPI(BaseModel):
    """Generic model for a single ESG Key Performance Indicator."""
    name: str = Field(description="Name of the KPI (e.g. 'GHG Intensity')")
    value: Union[str, float] = Field(description="Extracted value (NUMBER ONLY as string, e.g. '550', '22.5'). NO text/units here.")
    unit: str = Field(description="Unit of measurement (e.g. 'tCO2e', '%', 'TRIR').")
    year: str = Field(..., description="Fiscal year for this data point")
    trend: Optional[str] = Field(None, description="Explicitly stated trend (e.g. '-5%'). DO NOT CALCULATE.")
    standard_alignment: Optional[str] = Field(None, description="Standard used (e.g. 'GRI 305', 'SASB').")


class EnvironmentalPillar(BaseModel):
    """E-pillar metrics."""
    ghg_intensity: Optional[ESGKPI] = Field(None, description="E1. GHG INTENSITY (Scope 1+2+3)")
    renewable_energy: Optional[ESGKPI] = Field(None, description="E2. RENEWABLE ENERGY SHARE")


class SocialPillar(BaseModel):
    """S-pillar metrics."""
    trir: Optional[ESGKPI] = Field(None, description="S1. TRIR (Total Recordable Incident Rate)")
    women_in_leadership: Optional[ESGKPI] = Field(None, description="S2. WOMEN IN LEADERSHIP (%)")


class GovernancePillar(BaseModel):
    """G-pillar metrics."""
    supplier_esg_score: Optional[ESGKPI] = Field(None, description="G1. SUPPLIER ESG PERFORMANCE SCORE")
    traceability: Optional[ESGKPI] = Field(None, description="G2. SUPPLY CHAIN TRACEABILITY")


class ESGReport(BaseModel):
    """Complete ESG Data extracted from a report."""
    company_name: str = Field(default="Unknown", description="Name of the company")
    fiscal_year: str = Field(default="Unknown", description="Fiscal year of the report")
    environment: EnvironmentalPillar = Field(default_factory=EnvironmentalPillar)
    social: SocialPillar = Field(default_factory=SocialPillar)
    governance: GovernancePillar = Field(default_factory=GovernancePillar)
    extraction_confidence: float = Field(default=0.0, description="Confidence score of extraction (0-1)")
    
    # Methodological Metrics (Data Mining Quality)
    csr_density: float = Field(default=0.0, description="Proxy for information density (CSR Sentences / Total Sentences)")
    conciseness_proxy: float = Field(default=0.0, description="Proxy for text conciseness (KPIs extracted / Total Tokens)")
    
    # Token Accounting (Cost/Efficiency Awareness)
    initial_tokens: int = Field(default=0, description="Raw tokens before cleaning")
    final_tokens: int = Field(default=0, description="Tokens sent to LLM")
    token_reduction_pct: float = Field(default=0.0, description="Efficiency gain (%)")
    
    # AI Summary
    recap: Optional[str] = Field(None, description="Short AI-generated executive summary of the data coverage")
