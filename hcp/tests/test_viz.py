from numpy.testing import assert_equal

import hcp
from hcp.viz import make_hcp_bti_layout


def test_make_layout(hcp_params):
    """Test making a layout."""
    raw = hcp.read_raw(data_type="rest", **hcp_params).crop(0, 1).load_data()
    raw.pick_types(meg=True)
    lout = make_hcp_bti_layout(raw.info)
    assert_equal(lout.names, raw.info["ch_names"])
