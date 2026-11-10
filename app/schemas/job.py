from typing import Literal, Any
from pydantic import BaseModel, Field

class JobCreate(BaseModel):
    file_id: str
    operation: Literal["signature_detection"] = "signature_detection"

class SimilarityRequest(BaseModel):
    signature_a_file_id: str
    signature_b_file_id: str

class SimilarityResponse(BaseModel):
    similarity_score: float = Field(ge=0, le=1)
