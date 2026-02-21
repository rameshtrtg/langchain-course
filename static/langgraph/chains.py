from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an elite story writer. Generate best story in English based on the topic given by user."
        "The story should have around 100 words. "
        "If the user provides critique, respond with a revised version of your previous attempts without changing the original story.",
    ),
    MessagesPlaceholder(variable_name="messages")
])

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an editor who ensures that authors produce stories that fit the audience."
        "You ensure the story has content relevant to audience within 16 age group."
        "You ensure moral of the story is clear"
        "You check whether the story communicates only positive content."
        "You ensure that the author do not change the original story but just improve as per the feedback"
        "Always provide detailed recommendations, including requests for moral, age relevance and positivity",
    ),
    MessagesPlaceholder(variable_name="messages")
])

def setup_chains():
    llm = ChatOpenAI()
    generation_chain = generation_prompt | llm
    reflection_chain = reflection_prompt | llm
    return generation_chain, reflection_chain