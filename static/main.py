from dotenv import load_dotenv
import asyncio

from static.rag.web_source_rag import store_web_content_to_vector_db, tavily_extract_store_web_content_to_vector_db, \
    search_tamidas_document

load_dotenv()

async def main():

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
    search_tamidas_document()

if __name__ == "__main__":
    asyncio.run(main())
