import os
from dotenv import load_dotenv, find_dotenv
from workflows.nodes import generate_response, retrieve_documents, grade_documents, transform_query
from workflows.edges import decide_to_generate
from workflows.agents import document_grader, question_rewriter, answer_generator
from workflows.states import RAGState
from scripts.embedding_service import PineconeEmbeddingManager
from langgraph.graph import START, END, StateGraph


# load environment variables
load_dotenv(find_dotenv())
api_key = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('INDEX_NAME')
name_space = os.environ.get('NAMESPACE')

# instantiate the Pinecone Manager
manager = PineconeEmbeddingManager(api_key=api_key, index_name='kifiya', name_space='test')

# define the graph nodes
retriever = lambda state: retrieve_documents(state=state, retriever=manager)    
grader = lambda state: grade_documents(state=state, document_grader=document_grader)
rewriter = lambda state: transform_query(state=state, question_rewriter=question_rewriter)
generator = lambda state: generate_response(state=state, answer_generator=answer_generator)

# create a workflow/graph
workflow = StateGraph(RAGState)

# register the nodes to the workflow/graph
workflow.add_node("retrieve", retriever)
workflow.add_node("rewriter", rewriter)
workflow.add_node("grade_documents", grader)
workflow.add_node("answer_generator", generator)

# add edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("rewriter", "retrieve")
workflow.add_edge("answer_generator", END)

# add conditional edges
workflow.add_conditional_edges(source="grade_documents", path=decide_to_generate)

# compile the workflow/graph
app = workflow.compile()
