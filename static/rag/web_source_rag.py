import asyncio
import os
import ssl
from typing import List, Dict, Any
from operator import itemgetter

import certifi
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyMap, TavilyExtract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from static.common.logger import log_info, log_error


def setup_ssl_certificate():
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

def store_web_content_to_vector_db():
    setup_ssl_certificate()

    if 'TAVILY_API_KEY' not in os.environ:
        print("Provide TAVILY_API_KEY for this feature to work")
        return

    #Load
    log_info("Loading source data from tamidas website")
    tamidas_url = "https://www.tamidas.com"
    instruction = "Fetch all text content except contact, pricing and blog related content"
    tavily_crawl = TavilyCrawl()
    results = tavily_crawl.invoke({"url": tamidas_url, "instructions":instruction, "max_depth":5, "extract_depth":"advanced"})

    #split/transform
    log_info("Splitting/transforming source content for storing in vector db")
    crawled_data = results.get('results', [])

    documents = [
        Document(
            page_content=result['raw_content'],
            metadata={'source': result['url']}
        )
        for result in crawled_data
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    #Embed
    log_info("Preparing Embedding model for the split content")
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=50, retry_min_seconds=10)

    #Store
    log_info("Storing the data into Pinecone vector DB")
    vector_store = PineconeVectorStore(embedding=embedding, index_name="tamidas-web")
    vector_store.add_documents(docs)
    log_info("Completed Storing the data")

def chunk_url(urls:List[str], chunk_size: int = 3) -> List[List[str]]:
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunks.append(urls[i:i + chunk_size])
    return chunks

async def extract_batch(tavily_extract: TavilyExtract, urls: List[str], batch_numer:int) -> List[Dict[str,Any]]:
    result = await tavily_extract.ainvoke({"urls": urls})
    log_info(f"extracted data for batch: {batch_numer}")
    return result.get("results", [])

async def tavily_extract_store_web_content_to_vector_db():
    setup_ssl_certificate()

    if 'TAVILY_API_KEY' not in os.environ:
        print("Provide TAVILY_API_KEY for this feature to work")
        return

    #Load
    log_info("Loading source data from tamidas website")
    tamidas_url = "https://www.tamidas.com"
    tavily_map = TavilyMap()
    sitemap = tavily_map.invoke( tamidas_url)
    urls = sitemap.get("results", [])
    batched_urls = chunk_url(urls, chunk_size=3)
    tavily_extract = TavilyExtract()
    tasks = [extract_batch(tavily_extract, batch, i + 1) for i, batch in enumerate(batched_urls)]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    all_docs = []
    for result in batch_results:
        if isinstance(result, Exception):
            log_error(f"Batch failed with exception {result}")
        else:
            for extract in result:
                doc = Document(
                    page_content=extract['raw_content'],
                    metadata={'source': extract['url']}
                )
                all_docs.append(doc)
    #split/transform
    log_info("Splitting/transforming source content for storing in vector db")
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = splitter.split_documents(all_docs)

    #Embed
    log_info("Preparing Embedding model for the split content")
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=50, retry_min_seconds=10)

    #Store
    log_info("Storing the data into Pinecone vector DB")
    vector_store = PineconeVectorStore(embedding=embedding, index_name="tamidas-web")
    vector_store.add_documents(docs)
    log_info("Completed Storing the data")

def get_search_document_chain(retriever, prompt_template, llm):
    retrieval_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | retriever | format_docs
            )
            | prompt_template
            | llm
            | StrOutputParser()
    )
    return retrieval_chain

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstores = PineconeVectorStore(index_name="tamidas-web",
                                       embedding=embedding)
    retrieved_docs = vectorstores.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about Tamidas products. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    # Initialize chat model
    model = init_chat_model("gpt-5.2", model_provider="openai")
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }

def search_tamidas_document():
    """
        This is an example of Agentic RAG
        Workflow: User Goal -> Think (Reasoning) -> Act (Retrieve/Use Tool) -> Observe (Evaluate) -> Repeat or Final Answer.
    """
    # Query
    #question = "what kind of preventive maintenance support is available in TamidaS CMMS?"
    question = "what all products do TamidaS provide?"
    result = run_llm(question)
    print("\nAnswer:")
    print(result)