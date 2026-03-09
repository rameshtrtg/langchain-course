import os
from typing import Any, Dict

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from static.langgraph.corrective_rag.state import GraphState


def retrieve_doc(state: GraphState):
    print("---RETRIEVE---")
    question = state["question"]
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstores = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"),
                                       embedding=embedding)
    retriever = vectorstores.as_retriever(search_kwargs={"k": 3})
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}