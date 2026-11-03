import re
from decimal import Decimal

from datasets import Dataset, load_dataset


# Pin the source data so future downloads use the same revision.
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def extract_ground_truth(answer: str) -> Decimal:
    """Extract the numerical label after GSM8K's #### marker."""
    if answer.count("####") != 1:
        raise ValueError("Reference answer must contain exactly one #### marker.")

    final_answer = answer.split("####", 1)[1].strip()

    # Allow a sign, ordinary digits or grouped thousands, and a decimal part.
    # Checking the whole label rejects trailing text, NaN, and infinity too.
    number_pattern = r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?"
    if re.fullmatch(number_pattern, final_answer) is None:
        raise ValueError(f"Invalid numerical answer: {final_answer!r}")

    return Decimal(final_answer.replace(",", ""))


def load_development_data(sample_size: int = 30, seed: int = 42) -> Dataset:
    """Select reproducible development examples from GSM8K's training split."""
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("sample_size must be a positive integer.")

    train_data = load_dataset(
        "openai/gsm8k", "main", split="train", revision=GSM8K_REVISION
    )
    if sample_size > len(train_data):
        raise ValueError("sample_size cannot exceed the training split size.")

    # Add IDs before shuffling so each example keeps its original row number.
    train_data = train_data.add_column("source_index", list(range(len(train_data))))
    return train_data.shuffle(seed=seed).select(range(sample_size))


def main():
    dev_data = load_development_data()
    print("Dataset: openai/gsm8k, configuration: main, split: train")
    print(f"Revision: {GSM8K_REVISION}; shuffle seed: 42")
    print(f"Development questions: {len(dev_data)}")

    for i in range(min(5, len(dev_data))):
        example = dev_data[i]
        print(f"\n--- Example {i + 1} ---")
        print(f"Original training row: {example['source_index']}")
        print("QUESTION:")
        print(example["question"])
        print("REFERENCE ANSWER:")
        print(example["answer"])
        print("EXTRACTED LABEL:")
        print(extract_ground_truth(example["answer"]))


if __name__ == "__main__":
    main()
