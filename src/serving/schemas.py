from pydantic import BaseModel, Field
from typing import Optional, Dict

class LoanApplication(BaseModel):
    """
    Incoming payload for a new unbanked loan application.
    Notice that protected attributes (Gender, Caste) are explicitly MISSING
    from the production inference schema to prevent accidental direct bias.
    """
    borrower_id: str = Field(..., description="Anonymized Unique Identifier")
    
    # Financial/Cash-flow Signals
    inflow_consistency: float = Field(..., ge=0, le=1, description="Inflow stability index (0 to 1)")
    savings_buffer_days: float = Field(..., ge=0, description="Days of expenses covered by liquid funds")
    
    # Behavioral Signals
    phone_tenure_months: float = Field(..., ge=0, description="Months since first SIM registration")
    utility_timeliness: float = Field(..., ge=0, le=1, description="Percentage of on-time utility bills")
    
    # Loan parameters
    requested_amount: float = Field(..., gt=0, description="Requested loan principal")

class ScoringDecision(BaseModel):
    """
    The outbound response detailing the credit decision, including risk band,
    and threshold applied (for auditable governance).
    """
    borrower_id: str
    approved: bool
    risk_score: float = Field(..., ge=0, le=1)
    risk_band: str
    applied_threshold: float
    explanations: Dict[str, float] = Field(default_factory=dict, description="Local SHAP-like feature importances")
