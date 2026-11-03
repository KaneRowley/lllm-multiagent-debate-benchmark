import re
from decimal import Decimal
import os
from pathlib import Path
from load_data import extract_ground_truth, load_development_data
from dotenv import load_dotenv
from openai import OpenAI

def extract_model_answer(text):
    # Extract the final numerical answer from the model's response.
    match = re.search(
        r"^FINAL ANSWER:\s*([-+]?\d+(?:\.\d+)?)\s*$",
        text,
        flags=re.MULTILINE,
    )

    if match is None:
        return None

    return Decimal(match.group(1))


def main():
    # Load .env from the main project directory.
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    
    development_questions = load_development_data()
    question = development_questions[0]["question"]
    # Get the model name.
    model = os.getenv("MODEL_ID", "qwen3:4b")

    # Connect to Ollama's OpenAI-compatible API.
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Send a question to the model.

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Solve the following mathematics problem. "
                    "Explain your solution, then finish with a separate line "
                    "in this exact format:\n"
                    "FINAL ANSWER: <number>\n\n"
                    f"Problem:\n{question}"
                ),
            }
        ],
    )
    model_text = response.choices[0].message.content or ""
    model_answer = extract_model_answer(model_text)

    # Assumes "answer" contains GSM8K's original worked solution.
    reference_solution = development_questions[0]["answer"]
    correct_answer = Decimal(str(extract_ground_truth(reference_solution)))

    print("Model answer:", model_answer)
    print("Correct answer:", correct_answer)

    if model_answer is None:
        print("Could not extract the model's answer.")
    else:
        print("Correct:", model_answer == correct_answer)

    
    print("\nQuestion:\n")
    print(question)
    print("\nModel response:\n")
    print(response.choices[0].message.content)

    


if __name__ == "__main__":
    main()

