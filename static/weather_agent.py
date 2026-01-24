from langchain_tavily import TavilySearch
from langchain.agents import create_agent

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
