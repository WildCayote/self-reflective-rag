from workflows.states import RAGState

def decide_to_generate(state: RAGState, max_rewrites: int = 1) -> str:
    print("---INSPECT THE GRADED DOCUMENTS---")
    filtered_documents = state.get("documents", [])

    if not filtered_documents:
        rewrite_count = state.get("rewrite_count", 0)
        if rewrite_count >= max_rewrites:
            print("---DECISION: GENERATE REACHED MAX QUESTION REWRITE---")
            return "answer_generator"
        else:
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "rewriter"
    else:
        print("---DECISION: GENERATE---")
        return "answer_generator"

