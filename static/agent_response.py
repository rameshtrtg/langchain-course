from typing import List

from pydantic import BaseModel, Field

class Source(BaseModel):
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    answer:str = Field(description="The answers returned by the agent")
    sources:List[Source] = Field(default_factory=list, description="The list of sources used to generate the answer")


class AuthorResponse(BaseModel):
    short_summary:str = Field(description="The short summary")
    interesting_facts:List[str] = Field(description="Two Interesting Facts")

class AnswerResponse(BaseModel):
    answer:str = Field(description="The answers returned by the agent")
