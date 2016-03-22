# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import numpy as np
from mne.io import set_bipolar_reference
from mne.io.bti.bti import (
    _convert_coil_trans, _coil_trans_to_loc, _get_bti_dev_t,
    _loc_to_coil_trans)
from mne.transforms import Transform


def set_eog_ecg_channels(raw):
    """Set the HCP ECG and EOG channels

    Operates in place.

    Parameters
    ----------
    raw : instance of Raw
        the hcp raw data.
    """
    for kind in ['ECG', 'VEOG', 'HEOG']:
        set_bipolar_reference(
            raw, anode=kind + '-', cathode=kind + '+', ch_name=kind,
            copy=False)
    raw.set_channel_types({'ECG': 'ecg', 'VEOG': 'eog', 'HEOG': 'eog'})


def apply_ica_hcp(raw, ica_mat, exclude):
    """ Apply the HCP ICA.

    Operates in place.

    Parameters
    ----------
    raw : instance of Raw
        the hcp raw data.
    ica_mat : numpy structured array
        The hcp ICA solution
    exclude : array-like
        the components to be excluded.
    """
    assert ica_mat['topolabel'].tolist().tolist() == raw.ch_names[:]

    unmixing_matrix = np.array(ica_mat['unmixing'].tolist())

    n_components, n_channels = unmixing_matrix.shape
    mixing = np.array(ica_mat['topo'].tolist())

    proj_mat = (np.eye(n_channels) - np.dot(
        mixing[:, exclude], unmixing_matrix[exclude]))
    raw._data *= 1e15
    raw._data[:] = np.dot(proj_mat, raw._data)
    raw._data /= 1e15


def transform_sensors_to_mne(inst):
    """ Transform sensors to MNE coordinates

    For several reasons we do not use the MNE coordinates for the inverse
    modeling. This however won't always play nicely with visualization.

    """
    bti_dev_t = Transform('ctf_meg', 'meg', _get_bti_dev_t())
    dev_ctf_t = inst.info['dev_ctf_t']
    for ch in inst.info['chs']:
        loc = ch['loc'][:]
        if loc is not None:
            print('converting %s' % ch['ch_name'])
            t = _loc_to_coil_trans(loc)
            t = _convert_coil_trans(t, dev_ctf_t, bti_dev_t)
            loc = _coil_trans_to_loc(t)
            ch['loc'] = loc
