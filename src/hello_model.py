
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


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

    # Send a question to the model.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "A shop sells 17 boxes of pencils. "
                    "Each box contains 24 pencils. "
                    "How many pencils are there altogether?"
                ),
            }
        ],
    )

    print("\nModel response:\n")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()

