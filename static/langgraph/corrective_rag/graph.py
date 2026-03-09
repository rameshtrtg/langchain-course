from langchain_core.messages import AIMessage

from langgraph.graph import END, StateGraph

from static.langgraph.corrective_rag.constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from static.langgraph.corrective_rag.nodes import generate, grade_documents, retrieve, web_search
from static.langgraph.corrective_rag.state import GraphState


def decide_to_generate(state):
    """ Corrective RAG post retrieving the document for the user question,
    validates whether the retrieved document is good enough to answer the user question.
    If they are not relevant, it goes to web to search for answer.

    Corrective RAG -
     - 1. Guarantees the quality of the document that is used to answer the question
     - 2. It does not guaranty the quality of answer itself.
          For this you will need to go for Self-RAG implementation"""
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE

def run_corrective_rag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node(RETRIEVE, retrieve.retrieve_doc)
    workflow.add_node(GRADE_DOCUMENTS, grade_documents.grade_documents)
    workflow.add_node(GENERATE, generate.generate_answer)
    workflow.add_node(WEBSEARCH, web_search.web_search)

    workflow.set_entry_point(RETRIEVE)
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        decide_to_generate,
        {
            WEBSEARCH: WEBSEARCH,
            GENERATE: GENERATE,
        },
    )
    workflow.add_edge(WEBSEARCH, GENERATE)
    workflow.add_edge(GENERATE, END)

    graph = workflow.compile()

    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    # for relevant question checks in retrieved doc for answe
    res = graph.invoke(
        {
            "question": "Write about Tamidas products. Which product can be used for machine defect tracking?"
        }
    )
    # Extract the final answer from the last message with tool calls
    answer = res["generation"]
    print(answer)

    # for irrelevant question checks in retrieved web for answe
    res = graph.invoke(
        {
            "question": "What tool can be used for C# code generation"
        }
    )
    # Extract the final answer from the last message with tool calls
    answer = res["generation"]
    print(answer)