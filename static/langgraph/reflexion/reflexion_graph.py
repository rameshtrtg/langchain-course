from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from static.langgraph.reflexion.reflexion_chains import get_revisor_chain, get_first_responder_chain
from static.langgraph.reflexion.tool_executer import setup_tools

MAX_ITERATIONS = 2
DRAFT_NODE = "draft"
TOOL_EXECUTION_NODE = "execute_tools"
REVISE_NODE = "revise"

def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = get_first_responder_chain().invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = get_revisor_chain().invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal[TOOL_EXECUTION_NODE, END]:
    """Determine whether to continue or end based on iteration count."""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return TOOL_EXECUTION_NODE

def run_reflexion_graph():
    execute_tools = setup_tools()
    builder = StateGraph(MessagesState)
    builder.add_node(DRAFT_NODE, draft_node)
    builder.add_node(TOOL_EXECUTION_NODE, execute_tools)
    builder.add_node(REVISE_NODE, revise_node)
    builder.add_edge(START, DRAFT_NODE)
    builder.add_edge(DRAFT_NODE, TOOL_EXECUTION_NODE)
    builder.add_edge(TOOL_EXECUTION_NODE, REVISE_NODE)
    builder.add_conditional_edges(REVISE_NODE, event_loop, [TOOL_EXECUTION_NODE, END])
    graph = builder.compile()

    print(graph.get_graph().draw_mermaid())

    res = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write about Tamidas products. Which product can be used for machine defect tracking?",
                }
            ]
        }
    )
    # Extract the final answer from the last message with tool calls
    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(last_message.tool_calls[0]["args"]["answer"])
    print(res)
