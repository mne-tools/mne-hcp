# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import pytest
from pathlib import Path


def pytest_configure(config):
    """Configure pytest options."""
    # Markers
    for marker in (
        "slow",
    ):
        config.addinivalue_line("markers", marker)

    # Treat warnings as errors, plus an allowlist
    warning_lines = f"error::"
    warning_lines += r"""
    ignore:.*is non-interactive.*:UserWarning
    """
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)


@pytest.fixture(autouse=True)
def mpl_close():
    """Close all matplotlib windows after each test."""
    import matplotlib
    matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt
    yield
    plt.close("all")


@pytest.fixture()
def hcp_params():
    """The MNE-HCP hcp params."""
    return dict(
        hcp_path=Path("~/mne-hcp-data/mne-hcp-testing").expanduser().resolve(strict=True),
        subject="105923",
    )
