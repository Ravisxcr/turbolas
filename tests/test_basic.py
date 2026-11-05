"""Basic test suite for TurboLAS package setup."""

import turbolas


def test_package_version() -> None:
    """Ensure the package exposes a valid version string."""
    assert turbolas.__version__ is not None
    assert isinstance(turbolas.__version__, str)
