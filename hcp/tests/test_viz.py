from numpy.testing import assert_equal

import mne
import hcp
from hcp.tests import config as tconf
from hcp.viz import make_hcp_bti_layout  # , plot_coregistration

hcp_params = dict(hcp_path=tconf.hcp_path,
                  subject=tconf.test_subject)


def test_make_layout():
    """Test making a layout."""
    raw = hcp.read_raw(data_type='rest', **hcp_params).crop(0, 1).load_data()
    raw.pick_types()
    lout = make_hcp_bti_layout(raw.info)
    assert_equal(lout.names, raw.info['ch_names'])


def test_plot_coregistration():
    """Test plotting of coregistration."""
    # plot_coregistration(subjects_dir=tconf.subjects_dir)
    pass

mne.utils.run_tests_if_main()
