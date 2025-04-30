from typing import Optional
from pydantic import BaseModel, Field

class RAGRequest(BaseModel):
    user_id: Optional[str] = None
    question: str = Field(
        description="The question to be answered by the RAG"
    )

class User(BaseModel):
    id: str
    name: str

class UserIn(BaseModel):
    id: str