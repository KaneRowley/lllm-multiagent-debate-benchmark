import re
import time
from decimal import Decimal
import os
from pathlib import Path
from load_data import extract_ground_truth, load_development_data
from dotenv import load_dotenv
from openai import OpenAI

def extract_model_answer(text):
    match = re.search(
        r"^FINAL ANSWER:[ \t]*(?:"
        r"([-+]?\d+(?:\.\d+)?)"
        r"|<([-+]?\d+(?:\.\d+)?)>"
        r")[ \t]*$",
        text,
        flags=re.MULTILINE,
    )

    if match is None:
        return None

    number = match.group(1) or match.group(2)
    return Decimal(number)


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
        print(f"\n--- Question ID: gsm8k/train/{example['source_index']} ---")
        start = time.perf_counter()
        # Send a question to the model.
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": ("Explain your solution, then finish with a separate line "
                                "containing FINAL ANSWER: followed by the number only. "
                                "Do not include angle brackets, units, or formatting on that line.\n"
                                "Example: FINAL ANSWER: 72\n\n"
                                f"Problem:\n{question}"
                    ),
                }
            ],
        )
        elapsed = time.perf_counter() - start
        print(f"Response time: {elapsed:.2f} seconds")
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

