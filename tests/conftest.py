"""
Pytest configuration and shared fixtures for all tests

This file is automatically loaded by pytest and provides
fixtures and configuration that are shared across all test modules.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def reset_random_seed():
    """
    Automatically reset random seed before each test for reproducibility
    This ensures tests are independent and deterministic
    """
    np.random.seed(42)
    yield
    # Cleanup after test (if needed)


@pytest.fixture
def temp_data_dir(tmp_path):
    """
    Provide a temporary directory for test data
    Automatically cleaned up after test
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


def pytest_configure(config):
    """
    Additional pytest configuration
    """
    # You can add custom configuration here
    pass


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically
    """
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "Integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark physics validation tests
        if "physical" in item.nodeid.lower() or "Physical" in item.nodeid:
            item.add_marker(pytest.mark.physics)

        # Mark slow tests (large systems)
        if "large" in item.nodeid.lower() or "10000" in item.nodeid:
            item.add_marker(pytest.mark.slow)
