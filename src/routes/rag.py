from fastapi import APIRouter
from dtos.rag import RAGRequest
from workflows.graphs import app


rag_router = APIRouter(prefix='/rag', tags=['RAG'])

@rag_router.post("/query")
async def get_response(request: RAGRequest):
    result = app.invoke(
        {"question" : request.question}
    )
    
    return result 
