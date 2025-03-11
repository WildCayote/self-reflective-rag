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
            "prompt": str(prompt),
            "document": str(doc)
        })

        result = score
        print(result)
        if "'yes'" in result or "'Yes'" in result or "'YES'" in result or "yes" in result:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    return {"documents": filtered_docs, "prompt": prompt}

def generate_response(state: RAGState, answer_generator: ChatOpenAI) -> RAGState:
    print('---GENERATING RESPONSE---')
    documents = state["documents"]
    try:
        _  = state['rewrite_count']
        prompt = state['rewritten_prompt']

    except Exception as e:
        prompt = state['prompt']

    result = answer_generator.invoke({
        "question": prompt,
        "context": documents
    })

    return {
        "generation": result
    }

def transform_query(state: RAGState, prompt_rewriter: ChatOpenAI) -> RAGState:
    print('---REWRITTING---')
   
    try:
        count  = state['rewrite_count']
        prompt = state['rewritten_prompt']
        result = prompt_rewriter.invoke({
        "prompt": prompt
        })
        count += 1
    except Exception as e:
        prompt = state['prompt']
        result = prompt_rewriter.invoke({
        "prompt": prompt
        })
        count = 1

    return {"rewritten_prompt": result, "rewrite_count": count}

def generate_assistant_response(state: RAGState, assistant: ChatOpenAI) -> RAGState:
    print('---GENERATING ASSISTANT RESPONSE---')
    rag_generation = state['generation']
    conversation_history = state["conversation_history"]
    original_prompt = state["prompt"]

    result = assistant.invoke({
        "generation": rag_generation,
        "conversation_history": conversation_history,
        "prompt": original_prompt
    })

    return {
        "generation": result
    }