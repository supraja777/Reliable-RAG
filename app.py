from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLm-L6-v2")

# Docs to index
urls = [
    "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

chunk_size = 1000 
chunk_overlap = 200

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = len
    )

doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents = doc_splits,
    collection_name = "rag",
    embedding = embeddings
)

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {'k' : 4}
)


question = "What are the different kinds of agentic design patterns"
docs = retriever.invoke(question)

print(f"Title: {docs[0].metadata['title']}\n\nSource: {docs[0].metadata['source']}\n\nContent: {docs[0].page_content}\n")

# Check document relevancy

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# Data Model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""

    binary_score: str = Field(
        description = "Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatGroq(model="llama-3.1-8b-instant", temperature = 0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

#Prompt

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""


grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document : \n \n {document} \n \n User question : {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

docs_to_use = []

for doc in docs:
    print(doc.page_content, '\n', '-' * 50)
    res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
    print(res, '\n')
    if res.binary_score == 'yes':
        docs_to_use.append(doc)

# Generate Results 

from langchain_core.output_parsers import StrOutputParser

system = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. 
Use three-to-five sentences maximum and keep the answer concise."""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved documents : \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>")
    ]
)

# Post-processing
def format_docs(docs):
    return "\n".join(f"<doc{i+1}>:\nTitle:{doc.metadata['title']}\nSource:{doc.metadata['source']}\nContent:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))

rag_chain = rag_prompt | llm | StrOutputParser()

generation = rag_chain.invoke({"documents" : format_docs(docs_to_use), "question" : question})

print(generation)

# Check for hallucinations

# Data model

class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description = "Answer is grounded in facts 'yes' or 'no'"
    )

structured_llm_grader_for_hallucinations = llm.with_structured_output(GradeHallucinations)

hallucination_prompt = PromptTemplate(
        input_variables = ["documents", "generation"],
        template = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
            
            set of facts : {documents}
            LLM generated response : {generation}
        """
    )

llm_hallucination_chain = hallucination_prompt | structured_llm_grader

input_data = {"documents" : docs_to_use, "generation" : generation}
response = llm_hallucination_chain.invoke(input_data)

print(response)

