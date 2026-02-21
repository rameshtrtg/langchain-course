from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableSerializable
from langgraph.graph import END, StateGraph

from static.langgraph.message_graph import MessageGraph
from static.langgraph.chains import *

generative_chain: Optional[RunnableSerializable] = None
reflection_chain: Optional[RunnableSerializable] =None
GENNODE = "GenNode"
REFLECTIONNODE = "ReflectionNode"

def generative_node(state: MessageGraph):
    return {"messages": [generative_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    result = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=result.content)]}

def should_continue(state: MessageGraph):
    print(len(state["messages"]))
    if len(state["messages"]) > 6:
        return END
    return REFLECTIONNODE

def write_story():

    global generative_chain, reflection_chain
    generative_chain, reflection_chain = setup_chains()
    graph_builder = StateGraph(state_schema=MessageGraph)
    graph_builder.add_node(GENNODE, generative_node)
    graph_builder.add_node(REFLECTIONNODE, reflection_node)
    graph_builder.set_entry_point(GENNODE)

    graph_builder.add_conditional_edges(GENNODE, should_continue, {END: END, REFLECTIONNODE: REFLECTIONNODE})
    graph_builder.add_edge(REFLECTIONNODE, GENNODE)

    graph = graph_builder.compile()
    print(graph.get_graph().draw_mermaid())
    inputs = {
        "messages": [
            HumanMessage(
                content="""Write a story on topic "Greed"
                """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)

