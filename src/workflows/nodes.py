from scripts.embedding_service import PineconeEmbeddingManager
from workflows.states import RAGState
from langchain_openai import ChatOpenAI

def retrieve_documents(state: RAGState, retriever: PineconeEmbeddingManager) -> RAGState:
    print('--RETRIEVING--')
    question = state['question']
    documents = retriever.search_matching(query=question)

    return {"question": question, "documents": documents}

def grade_documents(state: RAGState, document_grader: ChatOpenAI) -> RAGState:
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state['question']
    documents = state['documents']

    filtered_docs = []
    for doc in documents:
        score = document_grader.invoke({
            "question": question,
            "document": doc
        })

        result = score
        if result == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    return {"documents": filtered_docs, "question": question}

def generate_response(state: RAGState) -> RAGState:
    ...

def transform_query(state: RAGState) -> RAGState:
    ...