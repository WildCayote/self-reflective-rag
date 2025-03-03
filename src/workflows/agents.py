import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from workflows.prompts import HALLUCINATION_PROMPT, ANSWER_PROMPT, REWRITER_PROMPT, RAG_PROMPT, GRADER_PROMPT
from workflows.models import GradeAnswer, GradeHallucinations, GradeDocuments


load_dotenv(find_dotenv())

llm = ChatOpenAI(
    base_url=os.environ.get('BASE_URI'),
    api_key=os.environ.get('API_KEY'),
    model=os.environ.get('MODEL_NAME'),
    temperature=0
)

# create an llm that will check if there are hallucinations
structured_hallucination_grader = llm.with_structured_output(GradeHallucinations)
hallucination_grader = HALLUCINATION_PROMPT | StrOutputParser() # Didn't try to enfornce structured output because deepseek doesn't support it

# create an llm that will grade if the answer is enough or not
structured_answer_grader = llm.with_structured_output(GradeAnswer)
answer_grader = ANSWER_PROMPT | StrOutputParser() # Didn't try to enfornce structured output because deepseek doesn't support it

# create an llm that will rewrite the question/user-prompt
question_rewriter = REWRITER_PROMPT | llm | StrOutputParser()

# create an llm that will grade the documents
structured_document_grader = llm.with_structured_output(GradeDocuments)
document_grader = GRADER_PROMPT | llm | StrOutputParser() # Didn't try to enfornce structured output because deepseek doesn't support it

# create an llm that will produce the final answer
answer_generator = RAG_PROMPT | llm | StrOutputParser()
