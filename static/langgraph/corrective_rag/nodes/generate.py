from typing import Any, Dict

from static.langgraph.corrective_rag.chains.generation import  get_generation_chain
from static.langgraph.corrective_rag.state import GraphState


def generate_answer(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = get_generation_chain().invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}