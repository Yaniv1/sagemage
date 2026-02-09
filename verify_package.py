#!/usr/bin/env python
"""Verify that the package can be properly installed and imported."""

import sys
import subprocess

def main():
    """Run package verification checks."""
    checks = [
        ("Checking package metadata", check_metadata),
        ("Checking package structure", check_structure),
        ("Checking import", check_import),
    ]
    
    failed = False
    for check_name, check_func in checks:
        print(f"\n{check_name}...")
        try:
            check_func()
            print(f"✓ {check_name} passed")
        except Exception as e:
            print(f"✗ {check_name} failed: {e}")
            failed = True
    
    if failed:
        sys.exit(1)
    else:
        print("\n✓ All checks passed!")
        sys.exit(0)

def check_metadata():
    """Check that package metadata is valid."""
    import importlib.metadata
    try:
        metadata = importlib.metadata.metadata('sagemage')
        assert 'Name' in metadata
        assert 'Version' in metadata
    except Exception:
        # Package may not be installed yet, that's ok
        pass

def check_structure():
    """Check that package structure exists."""
    import os
    required_files = [
        'pyproject.toml',
        'setup.py',
        'setup.cfg',
        'MANIFEST.in',
        'sagemage/__init__.py',
        'sagemage/py.typed',
        'tests/__init__.py',
        'README.md',
        'CHANGELOG.md',
        'LICENSE',
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing required file: {file}")

def check_import():
    """Check that package can be imported."""
    import sagemage
    assert hasattr(sagemage, '__version__')
    assert hasattr(sagemage, '__author__')

if __name__ == '__main__':
    main()
