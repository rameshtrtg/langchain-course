import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from static.langgraph.reflexion.reflexion_models import AnswerQuestion, ReviseAnswer

def get_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are expert researcher.
    Current time: {time}
    
    1. {first_instruction}
    2. Reflect and critique your answer. Be severe to maximize improvement.
    3. Recommend search queries to research information and improve your answer.""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Answer the user's question above using the required format."),
        ]
    ).partial(
        time=lambda: datetime.datetime.now().isoformat(),
    )

def get_first_responder_chain():
    #llm = ChatOpenAI(model="o4-mini")
    llm = ChatOpenAI(model="gpt-5")
    first_responder_prompt_template = get_prompt_template().partial(
        first_instruction="Provide a detailed ~250 word answer."
    )

    return first_responder_prompt_template | llm.bind_tools(
        tools=[AnswerQuestion], tool_choice="AnswerQuestion"
    )

def get_revisor_chain():
    #llm = ChatOpenAI(model="o4-mini")
    llm = ChatOpenAI(model="gpt-5")
    revise_instructions = """Revise your previous answer using the new information.
        - You should use the previous critique to add important information to your answer.
            - You MUST include numerical citations in your revised answer to ensure it can be verified.
            - Ensure to not hallucinate and answer should be grounded in retrieved documents.
            - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
                - [1] https://example.com
                - [2] https://example.com
        - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
    """

    return get_prompt_template().partial(
        first_instruction=revise_instructions
    ) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


