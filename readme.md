⚡ Reliable RAG Pipeline — High-Accuracy Retrieval-Augmented Generation
🧠 What is Reliable RAG?

Reliable RAG is an enhanced Retrieval-Augmented Generation pipeline designed to produce accurate, grounded, and trustworthy responses by combining:

High-quality embeddings

Strong document retrieval

LLM-based document relevance grading

Post-generation hallucination detection

Unlike traditional RAG—which simply retrieves chunks and feeds them to an LLM—Reliable RAG adds two layers of verification to ensure correctness at both the input and output stages.

It is ideal for:

Knowledge bases

Research assistants

Document Q&A

Legal, scientific, and enterprise contexts

Any application where hallucinations are unacceptable

🔍 Advantages Over Traditional RAG

Traditional RAG is powerful, but it suffers from several limitations:

Problem in Traditional RAG	How Reliable RAG Fixes It
Retrieves incorrect / semantically weak documents	LLM-based relevance grading filters irrelevant chunks
Allows hallucinated answers	Hallucination detection checks whether output is grounded
No guardrails on quality	Double evaluation: on retrieval + post-generation
Dependent solely on vector similarity	Adds LLM semantic validation for higher accuracy
No transparency in pipeline	Prints scores + documents at every stage
✔ Reliable RAG = Controlled + Verified + Accurate

By validating both retrieved input documents and final outputs, Reliable RAG significantly reduces errors and produces trustworthy responses consistently.

📘 Project: Reliable RAG with LangChain, Chroma, HuggingFace & Groq

This project implements a complete Reliable RAG pipeline using:

Web document ingestion (WebBaseLoader)

Text splitting (RecursiveCharacterTextSplitter)

HuggingFace MiniLM embeddings

Chroma vector database

Groq Llama 3.1 LLM for relevance grading

Groq hallucination checker for final answer validation

It demonstrates how to build a RAG system that is not just functional, but robust and dependable.

🚀 Features
1️⃣ Web Scraping & Document Loading

Pulls multiple articles from DeepLearning.ai’s "Agentic Design Patterns" series.

2️⃣ Smart Document Chunking

Uses:

chunk_size = 1000
chunk_overlap = 200


for contextual, retrieval-friendly splits.

3️⃣ High-Quality Embeddings

Using sentence-transformers/all-MiniLM-L6-v2, a fast and accurate embedding model.

4️⃣ Chroma Vector Store

Stores embedded documents for fast and scalable similarity search.

5️⃣ LLM-Based Relevance Grading

Each retrieved document is evaluated by a Groq LLM to ensure semantic relevance.

6️⃣ Final Answer Generation

Produces concise, grounded answers using Llama 3.1 (Groq).

7️⃣ Hallucination Detection

Checks whether the generated answer is grounded in the provided documents.

📦 Installation
pip install -r requirements.txt

🔧 Environment Setup

Create a .env file:

GROQ_API_KEY=your_api_key_here

🧪 Workflow 

Load URLs

Extract text

Split into chunks

Embed chunks

Store in Chroma

Retrieve relevant chunks

LLM grades each retrieved chunk

Filter out irrelevant ones

LLM generates a final answer

Hallucination check validates it

📝 Output 

Retrieved documents

Relevance scores

Final generated answer

Hallucination status

Transparent, structured, and verifiable.