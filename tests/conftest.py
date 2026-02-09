"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_fixture():
    """A sample pytest fixture."""
    return "test_data"
