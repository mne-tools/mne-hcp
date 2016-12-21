import os.path as op
import shutil

from nose.tools import assert_true
import matplotlib

import mne
from hcp.tests import config as tconf
from hcp import make_mne_anatomy, compute_forward_stack
from hcp.viz import plot_coregistration
from hcp.tests.config import expensive_test

matplotlib.use('Agg')

hcp_params = dict(hcp_path=tconf.hcp_path,
                  subject=tconf.test_subject)


@expensive_test
def test_anatomy():
    """Test anatomy functions (slow!)."""
    import matplotlib.pyplot as plt
    # This is where the data are after downloading from HCP
    temp_dir = mne.utils._TempDir()
    subjects_dir = op.join(temp_dir, 'hcp-subjects')
    recordings_path = op.join(temp_dir, 'hcp-meg')
    make_mne_anatomy(recordings_path=recordings_path,
                     subjects_dir=subjects_dir, **hcp_params)
    assert_true(op.isfile(op.join(subjects_dir, hcp_params['subject'], 'bem',
                                  'inner_skull.surf')))
    # Now we need fsaverage...
    mne_subjects_dir = mne.get_config('SUBJECTS_DIR')
    assert_true(mne_subjects_dir is not None)
    shutil.copytree(op.join(mne_subjects_dir, 'fsaverage'),
                    op.join(subjects_dir, 'fsaverage'))
    compute_forward_stack(
        subjects_dir=subjects_dir, recordings_path=recordings_path,
        src_params=dict(add_dist=False, spacing='oct1'),
        verbose=True, **hcp_params)
    # let's do our viz tests, too
    plot_coregistration(subjects_dir=subjects_dir,
                        recordings_path=recordings_path, **hcp_params)
    plt.close('all')
    mne.viz.plot_bem(subject=tconf.test_subject, subjects_dir=subjects_dir)
    plt.close('all')

mne.utils.run_tests_if_main()
