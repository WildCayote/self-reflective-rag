from scripts.embedding_service import PineconeEmbeddingManager
from workflows.states import RAGState

def retrieve_documents(state: RAGState, retriever: PineconeEmbeddingManager) -> RAGState:
    print('--RETRIEVING--')
    question = state['question']
    documents = retriever.search_matching(query=question)

    return {"question": question, "documents": documents}

def grade_documents(state: RAGState) -> RAGState:
    ...

def generate_response(state: RAGState) -> RAGState:
    ...

def transform_query(state: RAGState) -> RAGState:
    ...