
from langgraph.graph import END, StateGraph

from static.langgraph.self_rag.chains.answer_grader import get_answer_grader
from static.langgraph.self_rag.chains.hallucination_grader import get_hallucination_grader
from static.langgraph.corrective_rag.constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from static.langgraph.corrective_rag.nodes import generate, grade_documents, retrieve, web_search
from static.langgraph.corrective_rag.state import GraphState

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = get_hallucination_grader().invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = get_answer_grader().invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def run_self_rag():
    """Self RAG - self evaluates its own answer on 2 parts
        1. Whether the answer generated is resolves customer question
        2. Whether the answer generated is derived from the documents retrieved and not hallucinated
        The code example implements Corrective RAG to ensure we get right document and
        Self RAG to ensure we give relevant valid answer
        """
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

    graph.get_graph().draw_mermaid_png(output_file_path="self_rag_graph.png")
    res = graph.invoke(
            {
                "question": "What all products are Tamidas offering? In those, which product can be used for machine defect tracking with reason?"
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