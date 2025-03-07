from fastapi import APIRouter
from dtos.rag import RAGRequest
from workflows.graphs import app


rag_router = APIRouter(prefix='/rag', tags=['RAG'])

@rag_router.post("/query")
async def get_response(request: RAGRequest):
    if request.user_id != None:
        result = app.invoke(
        {
            "prompt" : request.question,
            "user_id": request.user_id
        }
        )
    else: 
        result = app.invoke(
            {"prompt" : request.question,}
        )
    
    return result 
