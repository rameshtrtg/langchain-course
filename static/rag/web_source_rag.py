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
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


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

def search_tamidas_document():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    #llm = ChatOpenAI()
    llm = ChatOllama(temperature=0, model="gemma3:4b")
    vectorstores = PineconeVectorStore(index_name="tamidas-web",
                                        embedding=embedding)
    retriever = vectorstores.as_retriever(search_kwargs={"k":3})
    prompt_template = ChatPromptTemplate.from_template(
   """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:""")

    # Query
    #question = "what kind of preventive maintenance support is available in TamidaS CMMS?"
    question = "what all products do TamidaS provide?"

    chain = get_search_document_chain(retriever,prompt_template,llm)
    result_with_lcel = chain.invoke({"question":question})
    print("\nAnswer:")
    print(result_with_lcel)