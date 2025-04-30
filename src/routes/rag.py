from fastapi import APIRouter
from dtos.rag import RAGRequest, User, UserIn
from workflows.nodes import query_extractor
from workflows.graphs import workflow
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime


# instantiate a short term memory checkpointer
memory = MemorySaver()
tmp_memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
app2 = workflow.compile(checkpointer=tmp_memory)


rag_router = APIRouter(prefix='/rag', tags=['RAG'])

@rag_router.post("/query")
async def get_response(request: RAGRequest):
    if request.user_id != None:
        config = {"configurable": {"thread_id": request.user_id}}
        
        # fetch the past messages
        state_snapshot = memory.get(config=config)
        history = []
        if state_snapshot:
            conversations = state_snapshot['channel_values']
            if 'conversation_history' in conversations:
                history.extend(conversations['conversation_history'])
            
            # add the prompt
            history.append(
                HumanMessage(content=conversations['prompt'])
            )

            # add the rag response
            history.append(
                AIMessage(content=conversations['generation'])
            )

        result = app.invoke(
        {
            "prompt" : request.question,
            "user_id": request.user_id,
            "conversation_history": history
        },
        config=config
        )
    else: 
        result = app.invoke(
            {"prompt" : request.question,}
        )
    
    return result 

@rag_router.post("/multi_query")
def get_multi_response(request: list[RAGRequest]):
    prompt = query_extractor(request)

    print(prompt)

    config = {"configurable": {"thread_id": 1}}
        
    # fetch the past messages
    state_snapshot = memory.get(config=config)
    history = []
    if state_snapshot:
        conversations = state_snapshot['channel_values']
        if 'conversation_history' in conversations:
            history.extend(conversations['conversation_history'])
        
        # add the prompt
        history.append(
            HumanMessage(content=conversations['prompt'])
        )

        # add the rag response
        history.append(
            AIMessage(content=conversations['generation'])
        )

    result = app.invoke(
    {
        "prompt" : prompt,
        "conversation_history": history
    },
    config=config
    )
    
      
    return result 




@rag_router.post("/greeting")
def greeting(User: UserIn):
    user = get_user(User)
    return greet_user(user)

def greet_user(user: User):
    # greet user based on the request time, morning, afternoon, or evening
    current_hour = datetime.now().hour
    if current_hour < 12:
        greeting = "Good morning"
    elif current_hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
    return {"message": f"{greeting}, {user.name}!"}
def get_user(user: UserIn):
    # Simulate a database lookup
    # In a real application, you would query your database here
    # For this example, we'll just return a dummy user
    user = User(id=user.id, name="John Doe")
    return user