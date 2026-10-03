from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

REQUIRED_FIELDS = ("TransactionDT", "TransactionAmt", "ProductCD", "card1")


class PredictRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, description="Caller-side id; used later by /feedback")
    fields: Dict[str, Optional[Union[float, str]]] = Field(
        ..., description="Raw transaction fields (IEEE-CIS names: TransactionDT, TransactionAmt, ProductCD, card1.., addr1, C*, D*, M*, V*, id_*, DeviceInfo ...)"
    )
    explain: bool = True
    commit: bool = Field(True, description="False = score without updating behaviour/entity state (what-if)")
    top_k: int = Field(5, ge=1, le=15)

    @field_validator("fields")
    @classmethod
    def _required(cls, v):
        missing = [k for k in REQUIRED_FIELDS if v.get(k) is None]
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        return v


class Reason(BaseModel):
    feature: str
    family: str
    value: Optional[float]
    shap: float
    direction: str
    text: str


class PredictResponse(BaseModel):
    transaction_id: str
    score: float
    calibrated_probability: float
    action: str
    reasons: List[Reason]
    flags: Dict[str, Any]
    model_version: str
    latency_ms: float


class FeedbackRequest(BaseModel):
    transaction_id: str
    is_fraud: bool


class FeedbackResponse(BaseModel):
    recorded: bool
    detail: str
