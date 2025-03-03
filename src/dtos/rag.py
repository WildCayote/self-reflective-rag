from pydantic import BaseModel, Field

class RAGRequest(BaseModel):
    question: str = Field(
        description="The question to be answered by the RAG"
    )
