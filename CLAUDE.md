# CLAUDE.md

## Commands
- **Install dependencies:** `pip install -r requirements.txt`
- **Run all tests:** `pytest`
- **Run specific test:** `pytest tests/test_filename.py`
- **Run analysis scripts:** `python analysis/script_name.py`
- **HPC Job Submission:** `sbatch scripts/script_name.sh` (or `bash scripts/script_name.sh` for local testing)

## Project Overview
This project investigates the spectral properties of the **Spectral Localiser** in computational physics. It implements tight-binding models to analyse level-statistics, specifically computing **r and z values**. It also compares the Spectral Localiser against the **Bianco-Resta** method for computing topological invariants.

## Architecture & Structure
- **Language:** Python (utilising NumPy, SciPy).
- **`/src/`:** Core model source code and logic.
- **`/scripts/`:** Execution scripts. Each Python script here has a corresponding `.sh` file for HPC execution.
- **`/notebooks/`:** Jupyter notebooks for interactive data analysis.
- **`/analysis/`:** Python scripts for heavy data analysis/post-processing.
- **`/tests/`:** Unit and regression tests.

## Coding Style & Conventions
- **Typing:** Use standard Python type hints for function arguments and return values.
- **Docstrings:** Use NumPy style docstrings for all physics functions.
- **Naming:** - Variables: `snake_case` (e.g., `hamiltonian_matrix`).
    - Classes: `PascalCase` (e.g., `SpectralLocaliser`).
- **Formatting:** Follow PEP 8.