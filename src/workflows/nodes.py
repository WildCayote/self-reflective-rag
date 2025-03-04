from scripts.embedding_service import PineconeEmbeddingManager
from workflows.states import RAGState
from langchain_openai import ChatOpenAI

def retrieve_documents(state: RAGState, retriever: PineconeEmbeddingManager) -> RAGState:
    print('---RETRIEVING---')
    prompt = state['prompt']
    documents = retriever.search_matching(query=prompt)

    return {"prompt": prompt, "documents": documents}

def grade_documents(state: RAGState, document_grader: ChatOpenAI) -> RAGState:
    print("---CHECK DOCUMENT RELEVANCE TO prompt---")
    prompt = state['prompt']
    documents = state['documents']

    filtered_docs = []
    for doc in documents:
        score = document_grader.invoke({
            "prompt": prompt,
            "document": doc
        })

        result = score.binary_score
        if result == 'yes':
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    return {"documents": filtered_docs, "prompt": prompt}

def generate_response(state: RAGState, answer_generator: ChatOpenAI) -> RAGState:
    print('---GENERATING RESPONSE---')
    prompt = state['prompt']
    documents = state["documents"]

    result = answer_generator.invoke({
        "question": prompt,
        "context": documents
    })

    return {
        "prompt": prompt,
        "documents": documents,
        "generation": result
    }

def transform_query(state: RAGState, prompt_rewriter: ChatOpenAI) -> RAGState:
    print('---REWRITTING---')
    prompt = state['prompt']
    
    result = prompt_rewriter.invoke({
        "prompt": prompt
    })

    try:
        count  = state['rewrite_count']
        count += 1
    except Exception as e:
        count = 1

    return {"prompt": result, "rewrite_count": count}

