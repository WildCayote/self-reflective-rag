from fastapi import APIRouter
from workflows.graphs import app


rag_router = APIRouter(prefix='/rag', tags=['RAG'])

@rag_router.post("/query")
async def get_response():
    return 'test' 
