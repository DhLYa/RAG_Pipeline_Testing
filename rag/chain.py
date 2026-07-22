from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
 
from .config import LLM_MODEL


def get_llm():
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    return llm

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template("""
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, say that you don't know.
    Do not use markdown formatting.
    Context: {context}
    Question: {question}
    """)

def build_chain(retriever) -> Runnable:
    chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | build_prompt()
    | get_llm()
    | StrOutputParser()
    )
    return chain

def ask(chain, question: str) -> str:
    return chain.invoke(question)
