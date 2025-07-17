import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser

# Load environment variables
load_dotenv()

# --- 1. Define the Final Output Schema ---
class FinalAnswer(BaseModel):
    """The final, structured answer for the user."""
    decision: str = Field(description="The final decision for the user, one of: 'Approved', 'Rejected', or 'More Info Needed'.")
    reasoning: str = Field(description="A clear, step-by-step explanation of how the decision was reached based on the provided documents and the user's query.")
    justification: str = Field(description="The single, exact, verbatim quote from the source documents that directly supports the decision. This must be a direct copy-paste from the provided context.")


# --- 2. Create the Reasoning Chain ---
def get_reasoning_chain():
    """
    Builds and returns a LangChain runnable that takes context and a query,
    and produces a final, structured answer.
    """
    # Initialize the Gemini chat model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Define the master prompt. This is the core of the reasoning engine.
    system_prompt = """
    You are an expert health insurance claims analyst. Your task is to provide a final, justified decision based on a user's query and relevant policy documents.

    **CONTEXT:**
    1.  **Parsed User Query:** This is the user's situation in a structured format.
    2.  **Policy Documents:** These are relevant excerpts from the insurance policy.

    **INSTRUCTIONS:**
    1.  **Analyze the User Query:** Carefully review the user's situation (age, procedure, etc.).
    2.  **Review Policy Documents:** Read the provided policy document excerpts to find the rules that apply to the user's query.
    3.  **Synthesize and Decide:** Compare the user's situation against the policy rules. Make a final `decision`: "Approved", "Rejected", or "More Info Needed".
        - **BE DECISIVE:** If the policy provides a clear rule (like an explicit exclusion, a waiting period, or a specific condition) that directly applies to the user's query, you MUST make a definitive 'Approved' or 'Rejected' decision. 
        - Only use 'More Info Needed' if a critical piece of information (e.g., a specific medical report) is truly missing from BOTH the query and the documents.
    4.  **Formulate Reasoning:** Write a clear, step-by-step `reasoning` that explains how you arrived at your decision, referencing the policy rules.
    5.  **Extract Justification:** This is the most critical step. Find the **single, most direct quote** from the "Policy Documents" that backs up your decision. The `justification` field MUST be an exact, verbatim copy-paste from the provided text.

    You must format your entire response as a single JSON object that strictly follows the schema of the `FinalAnswer` object.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        # The 'user' message will contain the structured query and retrieved docs
        ("user", "Parsed User Query:\n{parsed_query}\n\nPolicy Documents:\n{retrieved_docs_text}"),
    ])
    
    # Create a parser that ensures the output conforms to our Pydantic model
    output_parser = PydanticOutputParser(pydantic_object=FinalAnswer)

    # Combine the prompt, LLM, and output parser into a final chain
    return prompt | llm | output_parser


# --- 3. Test the Engine ---
if __name__ == "__main__":
    # Get the reasoning chain
    reasoning_chain = get_reasoning_chain()

    print("--- Testing Reasoning Engine ---")

    # Create mock data that simulates the output of our previous steps
    mock_parsed_query = {
        "procedure": "knee surgery",
        "plan_level": "Gold",
        "age": 46,
        "gender": "male",
        "location": "Mumbai"
    }

    mock_retrieved_docs_text = """
    Document 1:
    'Change-of-gender treatments: Expenses related to any treatment, including surgical management, to change characteristics of the body to those of the opposite sex are excluded.'
    
    Document 2:
    'Cosmetic or plastic Surgery: Expenses for cosmetic or plastic surgery or any treatment to change appearance unless for reconstruction following an Accident, Burn(s) or Cancer or as part of Medically Necessary Treatment.'
    
    Document 3:
    'Coverage for Joint Replacement Surgery, including knee replacement, is available under the Gold and Platinum plans for members over the age of 30. Pre-authorization is required at least 7 days before the scheduled procedure.'
    """

    # Invoke the chain with the mock data
    final_answer = reasoning_chain.invoke({
        "parsed_query": str(mock_parsed_query),
        "retrieved_docs_text": mock_retrieved_docs_text
    })

    print("\n--- Final Answer ---")
    print(f"Decision: {final_answer.decision}")
    print(f"Reasoning: {final_answer.reasoning}")
    print(f"Justification: '{final_answer.justification}'")