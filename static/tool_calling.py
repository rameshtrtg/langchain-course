from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage


from static.agent_response import AgentResponse


@tool
def get_text_length(text:str)->int:
    """Gets length of text"""
    print(f"input text: {text}")
    return len(text)

def tool_calling():
    llm = ChatOpenAI(model="gpt-5")
    tools = [get_text_length]
    agent = create_agent(model=llm, tools=tools)
    result = agent.invoke({"messages": HumanMessage(content="Given a question 'What is length of text DOG?', answer it in a sentence")})
    print(result)