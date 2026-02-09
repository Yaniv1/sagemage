"""Basic tests for sagemage package."""

import sagemage


def test_package_version():
    """Test that package has a version attribute."""
    assert hasattr(sagemage, "__version__")
    assert isinstance(sagemage.__version__, str)


def test_package_import():
    """Test that sagemage can be imported."""
    import sagemage

    assert sagemage is not None
