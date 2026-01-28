from langchain_tavily import TavilySearch
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from agent_response import *



def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def weather_agent():
    agent = create_agent(
        model="openai:gpt-5-mini",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
    )
    response = result.values()
    print(list(list(response)[-1])[-1].content)

@tool
def search_tool(query: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {query}!"

def weather_human_message():
    llm = ChatOpenAI(model="gpt-5")
    tools = [search_tool]
    agent = create_agent(model= llm, tools=tools)
    result = agent.invoke({"messages": HumanMessage(content="What is weather in Tokyo today?")})

@tool
def tavily_search_tool(query: str) -> dict:
    """Get weather for a given city."""
    tavily = TavilyClient()
    return tavily.search(query=query)

def weather_tavily_client():
    llm = ChatOpenAI(model="gpt-5")
    tools = [tavily_search_tool]
    agent = create_agent(model=llm, tools=tools)
    result = agent.invoke({"messages": HumanMessage(content="What is weather in Tokyo today?")})
    print(result["messages"][3].content)

def weather_tavily_search():
    llm = ChatOpenAI(model="gpt-5")
    tools = [TavilySearch(max_results=2, search_depth="basic")]
    # agent = create_agent(model=llm, tools=tools)
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
    result = agent.invoke({"messages": HumanMessage(content="What is weather in Tokyo today?")})
    print(result["structured_response"])

def person_info_tavily_search():
    prompt = f"""search about 'Ratan Naval Tata' from https://www.wikipedia.org and create
            1. A short summary
            2. two interesting facts about them
            """
    llm = ChatOpenAI(model="gpt-5")
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools, response_format=AuthorResponse)
    result = agent.invoke({"messages": HumanMessage(content=prompt)})
    print(result["structured_response"])