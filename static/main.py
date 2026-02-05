from dotenv import load_dotenv

from static.llm_ollama import gemma_chain
from static.tool_calling import *
from static.weather_agent import *

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
    pet_check_ollama()

if __name__ == "__main__":
    main()
