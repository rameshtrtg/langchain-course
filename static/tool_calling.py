from pyexpat.errors import messages

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool, BaseTool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from typing import List
from callbacks import AgentCallbackHandler



from static.agent_response import AgentResponse, AnswerResponse


@tool
def get_text_length(text:str)->int:
    """Gets length of text"""
    print(f"input text: {text}")
    return len(text)

def tool_calling():
    llm = ChatOpenAI(model="gpt-5")
    tools = [get_text_length]
    agent = create_agent(model=llm, tools=tools, response_format=AnswerResponse)
    result = agent.invoke({"messages": HumanMessage(content="Given a question 'What is length of text DOG?', answer it in a sentence")})
    print(result["structured_response"].answer)

@tool
def get_animal_category(animal:str) -> str:
    """given an animal name, gets category of animal"""
    if animal == "Pomeranian":
        return "Dog"
    if animal == "Pig":
        return "Free Animal"
    return "Wild Animal"

@tool
def can_pet(category:str) -> bool:
    """given category of the animal, gets whether we can pet the animal """
    if category == "Dog":
        return True
    return False

def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")

def pet_check_openai():
    llm = ChatOpenAI(model="gpt-5", temperature=0)
    tools = [get_animal_category, can_pet]
    agent = create_agent(model=llm, tools=tools,response_format=AnswerResponse)
    result = agent.invoke(
        {"messages": HumanMessage(content="Given a question 'Can we pet Pomeranian?', answer the question with reason")})
    print(result["structured_response"].answer)

def pet_check_ollama():
    llm = ChatOllama(model="gpt-oss:20b", temperature=0)
    #llm_with_structure = llm.with_structured_output(AnswerResponse)
    tools = [get_animal_category, can_pet]
    agent = create_agent(model=llm, tools=tools)
    result = agent.invoke(
        {"messages": HumanMessage(content="Given a question 'Can we pet Pomeranian?', answer the question with reason")})
    print(result["messages"][-1].content)

def pet_check_with_tool_calling_code():
    llm = ChatOllama(model="gpt-oss:20b", temperature=0, callbacks=[AgentCallbackHandler()])
    tools = [get_animal_category, can_pet]
    llm_with_tools = llm.bind_tools(tools)

    template = f"""Given a question 'Can we pet Pomeranian?', answer the question with reason"""

    messages = [HumanMessage(content=template)]
    while True:
        ai_message = llm_with_tools.invoke(messages)

        # If the model decides to call tools, execute them and return results
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                # tool_call is typically a dict with keys: id, type, name, args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            # Continue loop to allow the model to use the observations
            continue

        # No tool calls -> final answer
        print(ai_message.content)
        break