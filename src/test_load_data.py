from decimal import Decimal

import pytest
from datasets import Dataset

from src import load_data
from src.load_data import extract_ground_truth, load_development_data


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("Working...\n#### 72", Decimal("72")),
        ("Working...\n#### 72.0", Decimal("72")),
        ("Working...\n#### 1,234", Decimal("1234")),
        ("Working...\n#### -5", Decimal("-5")),
        ("Working...\n#### 2.5 \n", Decimal("2.5")),
        ("Working...\n#### 0", Decimal("0")),
        ("Working...\n#### +12", Decimal("12")),
        ("Working...\n#### -1,234.50", Decimal("-1234.50")),
        ("Working...\n#### 1,234,567", Decimal("1234567")),
    ],
)
def test_extract_ground_truth(answer, expected):
    assert extract_ground_truth(answer) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "No final answer marker",
        "Working...\n####",
        "Working...\n#### banana",
        "Working...\n#### NaN",
        "Working...\n#### Infinity",
        "Working...\n#### -Infinity",
        "Working...\n#### 1,2",
        "Working...\n#### 12,34",
        "Working...\n#### 1,,234",
        "Working...\n#### 72 pencils",
        "Working...\n#### 72\n#### 73",
        "Working...\n#### 1_000",
        "Working...\n#### 1e3",
    ],
)
def test_rejects_invalid_answers(answer):
    with pytest.raises(ValueError):
        extract_ground_truth(answer)


@pytest.fixture
def training_data(monkeypatch):
    """Replace the network download with a small dataset for unit tests."""
    data = Dataset.from_dict({
        "question": [f"Question {i}" for i in range(10)],
        "answer": [f"Working...\n#### {i}" for i in range(10)],
    })

    def fake_load_dataset(path, name, *, split, revision):
        assert (path, name, split) == ("openai/gsm8k", "main", "train")
        assert revision == load_data.GSM8K_REVISION
        return data

    monkeypatch.setattr(load_data, "load_dataset", fake_load_dataset)
    return data


def test_development_sample_is_reproducible(training_data):
    first = load_development_data(sample_size=4, seed=42)
    second = load_development_data(sample_size=4, seed=42)

    assert len(first) == 4
    assert first.to_dict() == second.to_dict()
    assert len(set(first["source_index"])) == 4
    for example in first:
        original = training_data[example["source_index"]]
        assert example["question"] == original["question"]
        assert example["answer"] == original["answer"]


@pytest.mark.parametrize("sample_size", [0, -1, 1.5, True, 11])
def test_rejects_invalid_sample_sizes(training_data, sample_size):
    with pytest.raises(ValueError):
        load_development_data(sample_size=sample_size)


def test_can_select_whole_training_split(training_data):
    sample = load_development_data(sample_size=len(training_data))
    assert sorted(sample["source_index"]) == list(range(len(training_data)))
