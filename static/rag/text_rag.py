import os
from operator import itemgetter
from os.path import ALLOW_MISSING

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



def search_text_data_indexing():
    loader = TextLoader(file_path="/Users/rameshramnath/Projects/Learning/ArtifitialIntelligence/LangChain/langchain-course/static/rag/tamidas.txt", autodetect_encoding=True)
    document = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="##")
    texts = splitter.split_documents(document)
    embedding = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"),model="text-embedding-3-large")
    PineconeVectorStore.from_documents(texts, embedding, index_name=os.environ.get("INDEX_NAME"))
    print('finished')

def get_search_text_chain(retriever, prompt_template, llm):
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
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

def search_text():
    """
        This is an example of 2-Step RAG.
        Workflow: User Query -> Retrieve Documents -> Generate Answer.
    """
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    # llm = ChatOpenAI()
    llm = ChatOllama(temperature=0, model="gemma3:4b")
    vectorstores = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"),
                                        embedding=embedding)
    retriever = vectorstores.as_retriever(search_kwargs={"k":3})
    prompt_template = ChatPromptTemplate.from_template(
   """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:""")

    # Query
    question = "what kind of preventive maintenance support is available in TamidaS CMMS?"
    chain = get_search_text_chain(retriever,prompt_template,llm)
    result_with_lcel = chain.invoke({"question":question})
    print("\nAnswer:")
    print(result_with_lcel)