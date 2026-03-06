from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def get_generation_chain():
    llm = ChatOpenAI(temperature=0)
    client = Client()

    prompt = client.pull_prompt("rlm/rag-prompt")
    print(prompt)
    return prompt | llm | StrOutputParser()