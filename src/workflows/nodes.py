from scripts.embedding_service import PineconeEmbeddingManager
from workflows.states import RAGState
from langchain_openai import ChatOpenAI

def retrieve_documents(state: RAGState, retriever: PineconeEmbeddingManager) -> RAGState:
    print('---RETRIEVING---')
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

def generate_response(state: RAGState, answer_generator: ChatOpenAI) -> RAGState:
    print('---GENERATING RESPONSE---')
    question = state['question']
    documents = state["documents"]

    result = answer_generator.invoke({
        "question": question,
        "context": documents
    })

    return {
        "question": question,
        "documents": documents,
        "generation": result
    }

def transform_query(state: RAGState, question_rewriter: ChatOpenAI) -> RAGState:
    print('---REWRITTING---')
    question = state['question']
    
    result = question_rewriter.invoke({
        "question": question
    })

    try:
        count  = state['rewrite_count']
        count += 1
    except Exception as e:
        count = 1

    return {"question": result, "rewrite_count": count}
