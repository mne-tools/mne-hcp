import warnings
from nose.tools import assert_true, assert_raises

import numpy as np
from numpy.testing import assert_equal

import mne
import hcp
from hcp.tests import config as tconf
from hcp.preprocessing import (apply_ref_correction, interpolate_missing,
                               map_ch_coords_to_mne, set_eog_ecg_channels,
                               apply_ica_hcp)

hcp_params = dict(hcp_path=tconf.hcp_path,
                  subject=tconf.test_subject)


def test_set_eog_ecg_channels():
    """Test setting of EOG and ECG channels."""
    raw = hcp.read_raw(data_type='rest', **hcp_params)
    raw.crop(0, 1).load_data()
    assert_equal(len(mne.pick_types(raw.info, meg=False, eog=True)), 0)
    assert_equal(len(mne.pick_types(raw.info, meg=False, ecg=True)), 13)
    set_eog_ecg_channels(raw)
    # XXX Probably shouldn't still have 8 ECG channels!
    assert_equal(len(mne.pick_types(raw.info, meg=False, eog=True)), 2)
    assert_equal(len(mne.pick_types(raw.info, meg=False, ecg=True)), 8)


def test_apply_ica():
    """Test ICA application."""
    raw = hcp.read_raw(data_type='rest', verbose='error', **hcp_params)
    annots = hcp.read_annot(data_type='rest', **hcp_params)
    # construct MNE annotations
    bad_seg = (annots['segments']['all']) / raw.info['sfreq']
    annotations = mne.Annotations(
        bad_seg[:, 0], (bad_seg[:, 1] - bad_seg[:, 0]), description='bad')

    raw.annotations = annotations
    raw.info['bads'].extend(annots['channels']['all'])
    ica_mat = hcp.read_ica(data_type='rest', **hcp_params)
    exclude = [ii for ii in range(annots['ica']['total_ic_number'][0])
               if ii not in annots['ica']['brain_ic_vs']]
    assert_raises(RuntimeError, apply_ica_hcp, raw, ica_mat=ica_mat,
                  exclude=exclude)
    # XXX right now this is just a smoke test, should really check some
    # values...
    with warnings.catch_warnings(record=True):
        raw.crop(0, 1).load_data()
    apply_ica_hcp(raw, ica_mat=ica_mat, exclude=exclude)


def test_apply_ref_correction():
    """Test reference correction."""
    raw = hcp.read_raw(data_type='rest', run_index=0, **hcp_params)
    # raw.crop(0, 10).load_data()
    raw.load_data()
    # XXX terrible hack to have more samples.
    # The test files are too short.
    raw.append(raw.copy())
    meg_picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    orig = raw[meg_picks[0]][0][0]
    apply_ref_correction(raw)
    proc = raw[meg_picks[0]][0][0]
    assert_true(np.linalg.norm(orig) > np.linalg.norm(proc))


def test_map_ch_coords_to_mne():
    """Test mapping of channel coords to MNE."""
    data_type = 'task_working_memory'
    hcp_evokeds = hcp.read_evokeds(onset='stim', data_type=data_type,
                                   **hcp_params)
    for evoked in hcp_evokeds:
        if evoked.comment == 'Wrkmem_LM-TIM-face_BT-diff_MODE-mag':
            break
    old_coord = evoked.info['chs'][0]['loc']
    map_ch_coords_to_mne(evoked)
    new_coord = evoked.info['chs'][0]['loc']
    assert_true((old_coord != new_coord).any())


def test_interpolate_missing():
    """Test interpolation of missing channels."""
    data_type = 'task_working_memory'
    raw = hcp.read_raw(data_type='task_working_memory', run_index=0,
                       **hcp_params)
    raw.load_data()
    n_chan = len(raw.ch_names)
    raw.drop_channels(['A1'])
    assert_equal(len(raw.ch_names), n_chan - 1)
    raw = interpolate_missing(raw, data_type=data_type, **hcp_params)
    assert_equal(len(raw.ch_names), n_chan)

    evoked = hcp.read_evokeds(data_type=data_type, **hcp_params)[0]
    assert_equal(len(evoked.ch_names), 243)
    evoked_int = interpolate_missing(evoked, data_type=data_type, **hcp_params)
    assert_equal(len(evoked_int.ch_names), 248)

mne.utils.run_tests_if_main()
