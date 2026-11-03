from decimal import Decimal

import pytest

from src.load_data import extract_ground_truth


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("Working...\n#### 72", Decimal("72")),
        ("Working...\n#### 72.0", Decimal("72")),
        ("Working...\n#### 1,234", Decimal("1234")),
        ("Working...\n#### -5", Decimal("-5")),
        ("Working...\n#### 2.5 \n", Decimal("2.5")),
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
    ],
)
def test_rejects_invalid_answers(answer):
    with pytest.raises(ValueError):
        extract_ground_truth(answer)