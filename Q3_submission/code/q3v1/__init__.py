"""Q3 competition-submission implementation."""

from .model import Q3Model
from .data import AlignedSample, TrainScaler, load_split, validate_content_indices
from .explain import exact_shapley, conditional_ig, summarize_contributions

__all__ = [
    "Q3Model",
    "AlignedSample",
    "TrainScaler",
    "load_split",
    "validate_content_indices",
    "exact_shapley",
    "conditional_ig",
    "summarize_contributions",
]
