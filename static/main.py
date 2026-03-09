import os

from dotenv import load_dotenv
import asyncio

from static.langgraph.corrective_rag.graph import run_corrective_rag_graph
from static.langgraph.reflection_story_writer import write_story
from static.langgraph.reflexion.reflexion_graph import run_reflexion_graph
from static.langgraph.self_rag.graph import run_self_rag
from static.rag.web_source_rag import store_web_content_to_vector_db, tavily_extract_store_web_content_to_vector_db, \
    search_tamidas_document

load_dotenv()

def main():

    # gemma_chain()
    # openai_gpt_chain()
    # weather_agent()
    # weather_tavily_client()
    # weather_tavily_search()
    # person_info_tavily_search()
    # tool_calling()
    #pet_check_openai()
    #pet_check_ollama()
    # search_text()
    #store_web_content_to_vector_db()
    #await tavily_extract_store_web_content_to_vector_db()
    #search_tamidas_document()
    #write_story()
    # run_reflexion_graph()
    # run_corrective_rag_graph()
    run_self_rag()

if __name__ == "__main__":
    main()
    #asyncio.run(main())
