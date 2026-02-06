#!/usr/bin/env python3
"""
TimingLLM Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="timingllm",
    version="0.1.0",
    author="Saher Elsayed",
    author_email="selsayed@seas.upenn.edu",
    description="LLM-Augmented FPGA Timing Closure Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saherelsayed/timingllm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "rag": [
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
        ],
        "llm": [
            "anthropic>=0.18.0",
            "openai>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "timingllm-generate=src.dataset_generator:main",
            "timingllm-experiment=experiments.run_experiments:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
