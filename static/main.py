from dotenv import load_dotenv

from static.llm_ollama import gemma_chain
from static.tool_calling import tool_calling
from static.weather_agent import *

load_dotenv()

def main():

    # gemma_chain()
    # openai_gpt_chain()
    # weather_agent()
    # weather_tavily_client()
    # weather_tavily_search()
    # person_info_tavily_search()
    tool_calling()

if __name__ == "__main__":
    main()
