from typing import List
from typing_extensions import TypedDict

class InputState(TypedDict):
    """
    Represents the state of the input sent by the user graph.

    Attributes:
        prompt: question
    """
    question: str

class IntermediateState(TypedDict):
    """
    Represents the intermediate state created during the RAG workflow that isn't of importance to the user

    Attributes:

    """

class RAGState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        prompt: question
        generation: LLM generation
        documents: list of documents
        rewrite_count: the number of rewrites of the query
    """

    prompt: str
    generation: str
    documents: List[str]
    rewrite_count: int
