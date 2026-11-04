import re
from decimal import Decimal
import os
from pathlib import Path
from src.load_data import extract_ground_truth, load_development_data
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
    
    # Get the model name.
    model = os.getenv("MODEL_ID", "qwen3:4b")

    # Connect to Ollama's OpenAI-compatible API.
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Keep track of the number of correct answers and parsing failures.
    correct_count = 0
    parsing_failures = 0
    development_questions = load_development_data()
    questions_to_run = development_questions.select(range(3))

    for example in questions_to_run:
        question = example["question"]
        
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
        reference_solution = example["answer"]
        correct_answer = extract_ground_truth(reference_solution)

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

        if model_answer is None:
            parsing_failures += 1
        elif model_answer == correct_answer:
            correct_count += 1

    total = len(questions_to_run)

    print(f"Correct: {correct_count}/{total}")
    print(f"Accuracy: {correct_count / total:.1%}")
    print(f"Parsing failures: {parsing_failures}")

    
if __name__ == "__main__":
    main()

