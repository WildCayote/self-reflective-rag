from fastapi import APIRouter
from dtos.rag import RAGRequest
from workflows.graphs import workflow
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.messages import HumanMessage, AIMessage
from scripts.chat_persistence_service import get_chat_history, save_chat_message

# instantiate a short term memory checkpointer
# memory = MemorySaver()
app = workflow.compile()


rag_router = APIRouter(prefix='/rag', tags=['RAG'])

@rag_router.post("/query")
async def get_response(request: RAGRequest):

    history = get_chat_history(request.user_id)

    result = app.invoke(
        {
            "prompt": request.question,
            "user_id": request.user_id,
            "conversation_history": history
        }
    )

    save_chat_message(request.user_id, request.question, result)

    return result


    # ai_message = "AI respponse to user question"
    # save_chat_message(request.user_id, request.question, ai_message)

    # if request.user_id != None:
    #     config = {"configurable": {"thread_id": request.user_id}}
        
    #     # fetch the past messages
    #     state_snapshot = memory.get(config=config)
    #     history = []
    #     if state_snapshot:
    #         conversations = state_snapshot['channel_values']
    #         if 'conversation_history' in conversations:
    #             history.extend(conversations['conversation_history'])
            
    #         # add the prompt
    #         history.append(
    #             HumanMessage(content=conversations['prompt'])
    #         )

    #         # add the rag response
    #         history.append(
    #             AIMessage(content=conversations['generation'])
    #         )

    #     result = app.invoke(
    #     {
    #         "prompt" : request.question,
    #         "user_id": request.user_id,
    #         "conversation_history": history
    #     },
    #     config=config
    #     )
    # else: 
    #     result = app.invoke(
    #         {"prompt" : request.question,}
    #     )
    
    # return result 
