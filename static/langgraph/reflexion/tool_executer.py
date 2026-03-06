import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

from static.langgraph.reflexion.reflexion_models import AnswerQuestion, ReviseAnswer

documents : list[str] = []
docids : list[str] = []

def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstores = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"),
                                        embedding=embedding)
    retriever = vectorstores.as_retriever(search_kwargs={"k":2})

    for search_query in search_queries:
        docs = retriever.invoke(search_query)
        for doc in docs:
            if doc.id not in docids:
                docids.append(doc)
                documents.append(doc.page_content)
    return documents

def setup_tools():
    return ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)