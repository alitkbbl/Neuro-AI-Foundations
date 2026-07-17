"""
create_project_structure.py

This script creates the initial directory and file structure for the
Neuro-AI-Foundations repository.

It is intentionally simple and safe:
- It creates missing directories.
- It creates missing files.
- It does NOT overwrite existing files by default.

Run:

    python create_project_structure.py
"""

from pathlib import Path


PROJECT_NAME = "Neuro-AI-Foundations"


def create_file_if_missing(path: Path, content: str = "") -> None:
    """
    Create a file only if it does not already exist.

    Parameters
    ----------
    path:
        Path to the file.

    content:
        Initial content to write into the file.
    """
    if path.exists():
        print(f"[SKIP] File already exists: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[CREATE] File: {path}")


def create_directory(path: Path) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path:
        Path to the directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directory: {path}")


def main() -> None:
    root = Path(PROJECT_NAME)

    # ------------------------------------------------------------------
    # Main directories
    # ------------------------------------------------------------------
    directories = [
        root,
        root / "src",
        root / "notebooks",
        root / "tests",
    ]

    for directory in directories:
        create_directory(directory)

    # ------------------------------------------------------------------
    # Root-level files
    # ------------------------------------------------------------------
    create_file_if_missing(
        root / "README.md",
        content="# Neuro-AI-Foundations\n\nInitial project scaffold.\n",
    )

    create_file_if_missing(
        root / "requirements.txt",
        content=(
            "numpy>=1.24\n"
            "scipy>=1.10\n"
            "matplotlib>=3.7\n"
            "jupyter>=1.0\n"
            "notebook>=7.0\n"
            "ipywidgets>=8.0\n"
            "nbformat>=5.9\n"
            "pytest>=7.0\n"
            "tqdm>=4.65\n"
        ),
    )

    create_file_if_missing(
        root / ".gitignore",
        content=(
            "# Python\n"
            "__pycache__/\n"
            "*.py[cod]\n"
            "*.pyo\n"
            "*.pyd\n"
            ".Python\n"
            "\n"
            "# Virtual environments\n"
            ".venv/\n"
            "venv/\n"
            "env/\n"
            "\n"
            "# Jupyter\n"
            ".ipynb_checkpoints/\n"
            "\n"
            "# Testing and coverage\n"
            ".pytest_cache/\n"
            ".coverage\n"
            "htmlcov/\n"
            "\n"
            "# OS files\n"
            ".DS_Store\n"
            "Thumbs.db\n"
        ),
    )

    # ------------------------------------------------------------------
    # Source files
    # ------------------------------------------------------------------
    create_file_if_missing(
        root / "src" / "__init__.py",
        content=(
            '"""\n'
            "Neuro-AI-Foundations source package.\n"
            '"""\n'
        ),
    )

    create_file_if_missing(
        root / "src" / "neuron_models.py",
        content=(
            '"""\n'
            "Neuron model implementations.\n\n"
            "Phase 1 starts with BaseNeuron.\n"
            "Later phases will add PassiveNeuron, LIFNeuron, and AdExNeuron.\n"
            '"""\n'
        ),
    )

    create_file_if_missing(
        root / "src" / "synapse_models.py",
        content=(
            '"""\n'
            "Synapse model implementations.\n\n"
            "This file will be completed in later phases.\n"
            '"""\n'
        ),
    )

    create_file_if_missing(
        root / "src" / "network_builder.py",
        content=(
            '"""\n'
            "Network construction and simulation utilities.\n\n"
            "This file will be completed in later phases.\n"
            '"""\n'
        ),
    )

    # ------------------------------------------------------------------
    # Notebook placeholders
    # ------------------------------------------------------------------
    notebook_placeholders = [
        "01_Passive_and_LIF.ipynb",
        "02_Non_Linear_Integrate_and_Fire.ipynb",
        "03_The_Need_for_Adaptation_AdEx.ipynb",
        "04_Phase_Plane_Analysis.ipynb",
        "05_Balanced_Recurrent_Network.ipynb",
    ]

    for notebook_name in notebook_placeholders:
        create_file_if_missing(
            root / "notebooks" / notebook_name,
            content="",
        )

    # ------------------------------------------------------------------
    # Test files
    # ------------------------------------------------------------------
    create_file_if_missing(
        root / "tests" / "__init__.py",
        content="",
    )

    create_file_if_missing(
        root / "tests" / "test_neuron_models.py",
        content=(
            '"""\n'
            "Unit tests for neuron models.\n\n"
            "Tests will be added as concrete neuron models are implemented.\n"
            '"""\n'
        ),
    )

    create_file_if_missing(
        root / "tests" / "test_synapse_models.py",
        content=(
            '"""\n'
            "Unit tests for synapse models.\n\n"
            "Tests will be added in later phases.\n"
            '"""\n'
        ),
    )

    print("\nProject structure created successfully.")
    print(f"Root directory: {root.resolve()}")


if __name__ == "__main__":
    main()
