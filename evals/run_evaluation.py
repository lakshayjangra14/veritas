import os
import json
import asyncio
import aiohttp
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
load_dotenv()
API_URL = "http://127.0.0.1:8000/query"
GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'golden_dataset.json')
# --- ADDED: Control concurrency to avoid rate limiting ---
MAX_CONCURRENT_REQUESTS = 3

# ... (The JustificationGrade and get_grading_chain functions remain the same) ...
class JustificationGrade(BaseModel):
    score: int = Field(description="A score from 1 to 5 for justification quality.")
    rationale: str = Field(description="A brief explanation for the score.")

def get_grading_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    parser = JsonOutputParser(pydantic_object=JustificationGrade)
    prompt = ChatPromptTemplate.from_template("""
    You are a strict grader. Evaluate the AI's justification based on the question and expected answer.
    CRITERIA: 1. Is it a real quote? 2. Does it support the decision?
    SCORING: 1=Poor, 5=Excellent.
    INPUTS:
    - User Question: {question}
    - Expected Decision: {expected_decision}
    - AI's Decision: {actual_decision}
    - AI's Justification: {actual_justification}
    Provide a score and rationale. {format_instructions}
    """, partial_variables={"format_instructions": parser.get_format_instructions()})
    return prompt | llm | parser

async def fetch_and_evaluate(session, item, grading_chain, semaphore):
    """Fetches a single result from the API and evaluates it, respecting the semaphore."""
    # --- MODIFIED: Acquire semaphore before running ---
    async with semaphore:
        print(f"--- Starting test for: {item['question_id']} ---")
        question = item['question']
        expected = item['expected_answer']
        
        try:
            timeout = aiohttp.ClientTimeout(total=180)
            async with session.post(API_URL, json={"text": question}, timeout=timeout) as response:
                response.raise_for_status()
                actual = await response.json()
            
            decision_correct = 1 if actual['decision'] == expected['decision'] else 0
            
            grade = await grading_chain.ainvoke({
                "question": question, "expected_decision": expected['decision'],
                "actual_decision": actual['decision'], "actual_justification": actual['justification']
            })
            justification_score = grade['score']
            justification_rationale = grade['rationale']
            
            print(f"--- Finished test for: {item['question_id']} ---")
            return {
                "ID": item['question_id'], "Decision Correct": "✅" if decision_correct else "❌",
                "Expected Decision": expected['decision'], "Actual Decision": actual['decision'],
                "Justification Score": justification_score, "Justification Rationale": justification_rationale
            }
        except Exception as e:
            print(f"--- FAILED test for: {item['question_id']} | Reason: {e} ---")
            return {
                "ID": item['question_id'], "Decision Correct": "❌", "Expected Decision": expected['decision'],
                "Actual Decision": "ERROR", "Justification Score": 0, "Justification Rationale": str(e)
            }

async def main():
    """Main function to run the evaluation concurrently with a semaphore."""
    # ... (loading dataset remains the same) ...
    with open(GOLDEN_DATASET_PATH, 'r') as f:
        golden_dataset = json.load(f)

    grading_chain = get_grading_chain()
    # --- ADDED: Initialize the semaphore ---
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    print(f"--- Starting Concurrent Evaluation (max {MAX_CONCURRENT_REQUESTS} parallel requests) ---")
    async with aiohttp.ClientSession() as session:
        # --- MODIFIED: Pass semaphore to the tasks ---
        tasks = [fetch_and_evaluate(session, item, grading_chain, semaphore) for item in golden_dataset]
        results = await asyncio.gather(*tasks)

    # ... (The reporting part remains the same) ...
    print("\n\n--- EVALUATION COMPLETE ---")
    df = pd.DataFrame([res for res in results if res])
    decision_accuracy_metric = df['Decision Correct'].value_counts(normalize=True).get("✅", 0) * 100
    avg_justification_score = df['Justification Score'].mean()
    print("\n--- Summary Metrics ---")
    print(f"Decision Accuracy: {decision_accuracy_metric:.2f}%")
    print(f"Average Justification Score: {avg_justification_score:.2f} / 5.0")
    print("\n--- Detailed Results ---")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 60)
    print(df.to_string())

if __name__ == "__main__":
    asyncio.run(main())