from typing import List, Optional
from typing_extensions import TypedDict

class InputState(TypedDict):
    """
    Represents the state of the input sent by the user graph.

    Attributes:
        prompt: question
    """
    user_id: Optional[str]
    prompt: str

class IntermediateState(TypedDict):
    """
    Represents the intermediate state created during the RAG workflow that isn't of importance to the user

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

class OutputState(TypedDict):
    """
    REpresents the state of the ouput sent by the workflow

    Attributes:
        content: the LLM generated response
    """
    generation: str

class RAGState(InputState, IntermediateState, OutputState):
    """
    Represents the state of our graph.
    """

