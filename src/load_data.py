from datasets import load_dataset
from decimal import Decimal, InvalidOperation

def extract_ground_truth(answer: str) -> Decimal:
    """Extract the numerical label after GSM8K's #### marker."""
    if "####" not in answer:
        raise ValueError("Reference answer is missing the #### marker.")

    final_answer = answer.rsplit("####", 1)[1]
    final_answer = final_answer.strip().replace(",", "")

    try:
        value = Decimal(final_answer)
    except InvalidOperation as exc:
        raise ValueError(
            f"Invalid numerical answer: {final_answer!r}"
        ) from exc

    if not value.is_finite():
        raise ValueError("Reference answer must be a finite number.")

    return value


def main():
    train_data = load_dataset("openai/gsm8k", "main", split="train")
    dev_data = train_data.shuffle(seed=42).select(range(30))
    print(f"Development questions: {len(dev_data)}")

    for i in range(5):
        example = dev_data[i]


        print(f"\n--- Example {i + 1} ---")
        print("QUESTION:")
        print(example["question"])
        print("REFERENCE ANSWER:")
        print(example["answer"])
        print("EXTRACTED LABEL:")
        print(extract_ground_truth(example["answer"]))


if __name__ == "__main__":
    main()