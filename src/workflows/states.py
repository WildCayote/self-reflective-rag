from typing import List, Optional
from typing_extensions import TypedDict

class InputState(TypedDict):
    """
    Represents the state of the input sent by the user graph.

    Attributes:
        user_id: the user id 
        conversation_history: the previous conversations of the user
        prompt: question
    """
    user_id: Optional[str]
    conversation_history: Optional[List[str]]
    prompt: str

class IntermediateState(TypedDict):
    """
    Represents the intermediate state created during the RAG workflow that isn't of importance to the user

    Attributes:
        prompt: question
        rewritten_prompt: the prompt rewritten by the LLM
        generation: LLM generation
        documents: list of documents
        rewrite_count: the number of rewrites of the query
    """
    prompt: str
    rewritten_prompt: str
    generation: str
    documents: List[str]
    rewrite_count: int

class OutputState(TypedDict):
    """
    REpresents the state of the ouput sent by the workflow

    Attributes:
        generation: the LLM generated response
    """
    generation: str

class RAGState(InputState, IntermediateState, OutputState):
    """
    Represents the state of our graph.
    """

