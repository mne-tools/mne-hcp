# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import numpy as np
import mne
from mne.io import set_bipolar_reference
from mne.io.bti.bti import (
    _convert_coil_trans, _coil_trans_to_loc, _get_bti_dev_t,
    _loc_to_coil_trans)
from mne.transforms import Transform
from mne.utils import logger
from mne.viz.topomap import _find_topomap_coords

from .io import read_info_hcp
from .io.read import _hcp_pick_info


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
            logger.debug('converting %s' % ch['ch_name'])
            t = _loc_to_coil_trans(loc)
            t = _convert_coil_trans(t, dev_ctf_t, bti_dev_t)
            loc = _coil_trans_to_loc(t)
            ch['loc'] = loc


def interpolate_missing_channels(inst, subject, data_type, hcp_path,
                                 run_index=0, mode='fast'):
    """ Interpolate all MEG channels that are missing

    Gentle warning: this might require some memory.
    """
    try:
        info = read_info_hcp(
            subject=subject, data_type=data_type, hcp_path=hcp_path,
            run_index=run_index if run_index is None else run_index)
    except (ValueError, IOError):
        raise ValueError(
            'could not find config to complete info.'
            'reading only channel positions without '
            'transforms.')
    # figure out which channels are missing
    bti_channel_names_ = ['A%i' % ii for ii in range(1, 249, 1)]
    bti_channel_names = [ch for ch in info['ch_names']
                         if ch in bti_channel_names_]
    fake_channels_to_add = sorted(
        list(set(bti_channel_names) - set(inst.ch_names)))
    n_channels = 248
    info['sfreq'] = inst.info['sfreq']
    # compute shape of data to be added
    if isinstance(inst, (mne.io.Raw,
                         mne.io.bti.bti.RawBTi)):
        shape = (n_channels,
                 (inst.last_samp - inst.first_samp) + 1)
        data = inst._data
    elif isinstance(inst, mne.Epochs):
        shape = (n_channels, len(inst.events), len(inst.times))
        data = inst.get_data()
    elif isinstance(inst, mne.Evoked):
        shape = (n_channels, len(inst.times))
        data = inst.data
    else:
        raise ValueError('instance must be Raw, Epochs '
                         'or Evoked')

    # create new data
    existing_channels_index = [
        bti_channel_names.index(ch) for ch in inst.ch_names]
    new_channels_index = [
        bti_channel_names.index(ch) for ch in fake_channels_to_add]

    out_data = np.empty(shape, dtype=np.float64)
    out_data[existing_channels_index] = data
    out_data[new_channels_index] = 0
    info = _hcp_pick_info(info, bti_channel_names)

    if isinstance(inst, (
            mne.io.Raw, mne.io.bti.bti.RawBTi)):
        out = mne.io.RawArray(out_data, info)
    elif isinstance(inst, mne.Epochs):
        out = mne.EpochsArray(data=out_data, info=info, eventds=inst.events,
                              tmin=inst.times.min(), event_id=inst.event_id)
    elif isinstance(inst, mne.Evoked):
        out = mne.EvokedArray(
            data=out_data, info=info, tmin=inst.times.min(),
            comment=inst.comment, nave=inst.nave, kind=inst.kind)
    else:
        raise ValueError('instance must be Raw, Epochs '
                         'or Evoked')

    # set "bad" channels and interpolate.
    out.info['bads'] = fake_channels_to_add
    out.interpolate_bads(mode=mode)
    return out


def make_hcp_bti_layout(info):
    """ Get Layout of HCP Magnes3600WH data """
    picks = list(range(248))
    pos = _find_topomap_coords(info, picks=picks)
    return mne.channels.layout.Layout(
        box=(-42.19, 43.52, -41.7, 28.71), pos=pos,
        names=[info['ch_names'][idx] for idx in picks], ids=picks,
        kind='magnesWH3600_hcp')
