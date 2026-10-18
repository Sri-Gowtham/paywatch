"""The UPI transaction contract and a pseudonymisation helper.

Identifiers (payer, payee, device) must be pseudonymised BEFORE they reach PayWatch. `pseudonymise` is a keyed hash: stable for the same
input and salt, not reversible without the salt. The schema rejects values that look like raw identifiers (a VPA contains '@', a phone
number is a long run of digits) so a configuration mistake fails loudly instead of leaking personal data into logs.
"""

import hashlib
import hmac
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

CHANNELS = ("app", "qr", "collect", "intent")
_RAW_PHONE = re.compile(r"^\+?\d{10,13}$")


def pseudonymise(value: str, salt: str) -> str:
    """Keyed hash (HMAC-SHA256, first 16 hex characters). Keep `salt` secret and constant; changing it changes every id."""
    if not salt:
        raise ValueError("a non-empty salt is required")
    return hmac.new(salt.encode("utf-8"), str(value).encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def looks_raw(value: str) -> bool:
    return "@" in value or bool(_RAW_PHONE.match(value))


class UpiTransaction(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=128, description="caller-side id, later used by /feedback")
    timestamp: float = Field(..., gt=0, description="transaction time, Unix seconds")
    amount: float = Field(..., gt=0, le=10_000_000, description="amount in rupees")
    payer_id: str = Field(..., min_length=1, max_length=128, description="PSEUDONYMISED payer identifier")
    payee_id: str = Field(..., min_length=1, max_length=128, description="PSEUDONYMISED payee identifier")
    device_id: Optional[str] = Field(None, max_length=128, description="PSEUDONYMISED device identifier, if known")
    channel: Literal["app", "qr", "collect", "intent"] = "app"
    payee_category: Optional[str] = Field(None, max_length=64, description="merchant category, if any")

    @field_validator("payer_id", "payee_id", "device_id")
    @classmethod
    def _must_be_pseudonymised(cls, v):
        if v is not None and looks_raw(v):
            raise ValueError("identifiers must be pseudonymised before they reach PayWatch (a raw VPA or phone number was supplied)")
        return v
