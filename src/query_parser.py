import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class ParsedQuery(BaseModel):
    """
    A structured representation of a user's query for an insurance policy.
    """
    procedure: Optional[str] = Field(
        None, description="The medical procedure or treatment the user is asking about (e.g., 'knee surgery', 'cancer treatment')."
    )
    plan_level: Optional[str] = Field(
        None, description="The specific insurance plan level mentioned (e.g., 'Gold', 'Silver', 'Basic')."
    )
    age: Optional[int] = Field(
        None, description="The age of the person in question."
    )
    gender: Optional[str] = Field(
        None, description="The gender of the person in question."
    )
    location: Optional[str] = Field(
        None, description="The geographical location or city mentioned."
    )


def get_query_parser_chain():
    """
    Builds and returns a LangChain runnable that parses a user query
    into a structured Pydantic object.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    structured_llm = llm.with_structured_output(ParsedQuery)

    system_prompt = """
    You are an expert at extracting key information from user queries related to health insurance.
    Your task is to parse the user's query and populate the fields of the 'ParsedQuery' object.
    
    - Only extract information that is explicitly mentioned in the query.
    - If a piece of information is not present, leave the corresponding field null.
    - For 'procedure', identify the primary medical service being asked about.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}"),
    ])

    return prompt | structured_llm

if __name__ == "__main__":
    parser = get_query_parser_chain()
    print("--- Testing Query Parser ---")
    query1 = "I am a 46 year old male who needs knee surgery. Am I covered under the Gold Plan in Mumbai?"
    result1 = parser.invoke({"query": query1})
    print(f"\nQuery: '{query1}'")
    print(f"Parsed: {result1}")

    query2 = "What's the coverage for cancer treatment?"
    result2 = parser.invoke({"query": query2})
    print(f"\nQuery: '{query2}'")
    print(f"Parsed: {result2}")
    
    query3 = "What is the policy on data privacy?"
    result3 = parser.invoke({"query": query3})
    print(f"\nQuery: '{query3}'")
    print(f"Parsed: {result3}")