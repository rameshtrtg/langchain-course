from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from static.langgraph.self_rag.graph import decide_to_generate, grade_generation_grounded_in_documents_and_question
from static.langgraph.adaptive_rag.chains.router import get_question_router, RouteQuery
from static.langgraph.corrective_rag.constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from static.langgraph.corrective_rag.nodes import generate, grade_documents, retrieve, web_search
from static.langgraph.corrective_rag.state import GraphState

load_dotenv()

def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = get_question_router().invoke({"question": question})
    print(source.datasource)
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

def run_adaptive_rag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node(RETRIEVE, retrieve.retrieve_doc)
    workflow.add_node(GRADE_DOCUMENTS, grade_documents.grade_documents)
    workflow.add_node(GENERATE, generate.generate_answer)
    workflow.add_node(WEBSEARCH, web_search.web_search)

    workflow.set_conditional_entry_point(
        route_question,
        {
            WEBSEARCH: WEBSEARCH,
            RETRIEVE: RETRIEVE,
        },
    )
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        decide_to_generate,
        {
            WEBSEARCH: WEBSEARCH,
            GENERATE: GENERATE,
        },
    )

    workflow.add_conditional_edges(
        GENERATE,
        grade_generation_grounded_in_documents_and_question,
        {
            "not supported": GENERATE,
            "useful": END,
            "not useful": WEBSEARCH,
        },
    )
    workflow.add_edge(WEBSEARCH, GENERATE)
    workflow.add_edge(GENERATE, END)

    graph = workflow.compile()

    graph.get_graph().draw_mermaid_png(output_file_path="adaptive_rag_graph.png")

    """res = graph.invoke(
                {
                    "question": "What all products are Tamidas offering? In those, which product can be used for machine defect tracking with reason?"
                }
            )
    # Extract the final answer from the last message with tool calls
    answer = res["generation"]
    print(answer)"""

    # for irrelevant question checks in retrieved web for answe
    res = graph.invoke(
        {
            "question": "What tool can be used for C# code generation"
        }
    )
    # Extract the final answer from the last message with tool calls
    answer = res["generation"]
    print(answer)
