"""
TimingLLM: LLM-Augmented FPGA Timing Closure
============================================

A framework for automated timing violation diagnosis and fix recommendation
using Large Language Models with Retrieval-Augmented Generation.

Author: Saher Elsayed
Institution: University of Pennsylvania
"""

from .dataset_generator import (
    generate_dataset,
    generate_design_timing_report,
    ViolationType,
    TimingPath,
    TimingReport,
)

from .timingllm_agent import (
    TimingLLMAgent,
    TimingKnowledgeBase,
    TimingViolationClassifier,
    DiagnosisResult,
    evaluate_agent,
)

from .utils import (
    load_dataset,
    save_results,
    extract_all_paths,
    split_dataset,
)

__version__ = "0.1.0"
__author__ = "Saher Elsayed"
__email__ = "selsayed@seas.upenn.edu"

__all__ = [
    # Dataset generation
    "generate_dataset",
    "generate_design_timing_report",
    "ViolationType",
    "TimingPath",
    "TimingReport",
    # Agent
    "TimingLLMAgent",
    "TimingKnowledgeBase",
    "TimingViolationClassifier",
    "DiagnosisResult",
    "evaluate_agent",
    # Utilities
    "load_dataset",
    "save_results",
    "extract_all_paths",
    "split_dataset",
]
