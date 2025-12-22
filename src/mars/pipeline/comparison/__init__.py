"""
Comparison module for MARS.

Compares exemplar and candidate databases to show incremental data recovery -
what NEW data was recovered from candidates that wasn't in the exemplar.
"""

from mars.pipeline.comparison.comparison_calculator import ComparisonCalculator
from mars.pipeline.comparison.html_report import generate_report
from mars.pipeline.comparison.types import (
    ComparisonResult,
    DatabaseComparison,
    LostAndFoundStats,
    TableComparison,
)

__all__ = [
    "ComparisonCalculator",
    "ComparisonResult",
    "DatabaseComparison",
    "LostAndFoundStats",
    "TableComparison",
    "generate_report",
]
