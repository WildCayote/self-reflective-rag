from typing import List
from typing_extensions import TypedDict

class RAGState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        rewrite_count: the number of rewrites of the query
    """

    question: str
    generation: str
    documents: List[str]
    rewrite_count: int
