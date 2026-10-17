"""Shared evaluation utilities (see unified.py and acoustic_degradation.py)."""

from .acoustic_degradation import (
    AcousticDegradationEngine,
    ParsedCondition,
    parse_condition_str,
    get_condition_metadata,
    get_acoustic_degradation_engine,
    CANONICAL_NOISE_TYPES,
    CANONICAL_SNR_LADDER,
    CANONICAL_EVAL_CONDITIONS,
    PARADIGM_A_CONDITIONS,
    PARADIGM_A_NOISE_TYPES,
    PARADIGM_A_SNR_LADDER,
)

__all__ = [
    "AcousticDegradationEngine",
    "ParsedCondition",
    "parse_condition_str",
    "get_condition_metadata",
    "get_acoustic_degradation_engine",
    "CANONICAL_NOISE_TYPES",
    "CANONICAL_SNR_LADDER",
    "CANONICAL_EVAL_CONDITIONS",
    "PARADIGM_A_CONDITIONS",
    "PARADIGM_A_NOISE_TYPES",
    "PARADIGM_A_SNR_LADDER",
]
