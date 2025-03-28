from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You a prompt re-writer that converts an input prompt to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. 
     \n DO NOT PUT IN YOUR REASONING. RETURN THE IMPROVED PROMPT YOU THINK WILL WORK NOT ANYTHIN MORE OR ANYTHIN LESS!"""),
        (
            "human",
            "Here is the initial prompt: \n\n {prompt} \n Formulate an improved prompt.",
        ),
    ]
)

GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing relevance of a retrieved document to a user prompt. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user prompt, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the prompt."""),
        ("human", "Retrieved document: \n\n {document} \n\n User prompt: {prompt}"),
    ]
)

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a grader assessing whether an answer addresses / resolves a prompt \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the prompt."""),
        ("human", "User prompt: \n\n {prompt} \n\n LLM generation: {generation}"),
    ]
)

RAG_PROMPT = hub.pull("rlm/rag-prompt")

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You are KAVAS, the AI assistant at Kifiya Technologies. Your responses must be:
1. Concise (2-4 sentences max) 
2. Precise (focus on key information from the RAG response)
3. Naturally human (use contractions, minimal pleasantries)
4. If you don't have the answer, do not mention the conversation history. just say I dont have enough information,

Format rules:
- Start directly with the answer
- Add ONLY ONE humanizing element (short question/empathy phrase)
- Never end with "feel free to ask" or similar open-ended yapping

Bad example: 
"Kifiya is innovative... *long paragraph*... How can I assist you further today?"

Good example: 
"Kifiya leads Africa's fintech innovation through mobile banking solutions. Their recent partnership with Safaricom expanded services to Kenya - impressive progress! Need specifics on their tech?"""),

        ("human", """RAG response:  
{generation}  

Conversation History:  
{conversation_history}  

User's current question:  
{prompt}  

Respond in MAX 4 SENTENCES:""")
    ]
)