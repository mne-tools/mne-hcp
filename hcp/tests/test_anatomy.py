import shutil

import mne
import pytest

from hcp import compute_forward_stack, make_mne_anatomy
from hcp.viz import plot_coregistration


@pytest.mark.slow
def test_anatomy(tmp_path, hcp_params):
    """Test anatomy functions (slow!)."""
    # This is where the data are after downloading from HCP
    subjects_dir = tmp_path / "hcp-subjects"
    recordings_path = tmp_path / "hcp-meg"
    make_mne_anatomy(
        recordings_path=recordings_path,
        subjects_dir=subjects_dir,
        verbose=True,
        **hcp_params,
    )
    subject_dir = subjects_dir / hcp_params["subject"]
    inner_skull = subject_dir / "bem" / "inner_skull.surf"
    assert inner_skull.is_file()
    white = subject_dir / "surf" / "lh.white"
    assert white.is_file()

    # Now we need fsaverage...
    mne_subjects_dir = mne.utils.get_subjects_dir(raise_error=True)
    shutil.copytree(mne_subjects_dir / "fsaverage", subjects_dir / "fsaverage")
    compute_forward_stack(
        subjects_dir=subjects_dir,
        recordings_path=recordings_path,
        src_params=dict(add_dist=False, spacing="oct1"),
        verbose=True,
        **hcp_params,
    )
    # let's do our viz tests, too
    plot_coregistration(
        subjects_dir=subjects_dir,
        recordings_path=recordings_path,
        **hcp_params,
    )
    mne.viz.plot_bem(subject=hcp_params["subject"], subjects_dir=subjects_dir)
