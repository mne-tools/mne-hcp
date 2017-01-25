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

from .io import read_info
from .io.read import _hcp_pick_info
from .io.read import _data_labels


def set_eog_ecg_channels(raw):
    """Set the HCP ECG and EOG channels

    .. note::
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
    """Apply the HCP ICA.

    .. note::
       Operates in place and data must be loaded.

    Parameters
    ----------
    raw : instance of Raw
        the hcp raw data.
    ica_mat : numpy structured array
        The hcp ICA solution
    exclude : array-like
        the components to be excluded.
    """
    if not raw.preload:
        raise RuntimeError('raw data must be loaded, use raw.load_data()')
    ch_names = ica_mat['topolabel'].tolist().tolist()
    picks = mne.pick_channels(raw.info['ch_names'], include=ch_names)
    assert ch_names == [raw.ch_names[p] for p in picks]

    unmixing_matrix = np.array(ica_mat['unmixing'].tolist())

    n_components, n_channels = unmixing_matrix.shape
    mixing = np.array(ica_mat['topo'].tolist())

    proj_mat = (np.eye(n_channels) - np.dot(
        mixing[:, exclude], unmixing_matrix[exclude]))
    raw._data *= 1e15
    raw._data[picks] = np.dot(proj_mat, raw._data[picks])
    raw._data /= 1e15


def apply_ref_correction(raw, decim_fit=100):
    """Regress out MEG ref channels

    Computes linear models from MEG reference channels
    on each sensors, predicts the MEG data and subtracts
    and computes the residual by subtracting the predictions.

    .. note::
       Operates in place.

    .. note::
       Can be memory demanding. To alleviate this problem the model can be fit
       on decimated data. This is legitimate because the linear model does
       not have any representation of time, only the distributions
       matter.

    Parameters
    ----------
    raw : instance of Raw
        The BTi/4D raw data.
    decim_fit : int
        The decimation factor used for fitting the model.
        Defaults to 100.
    """
    from sklearn.linear_model import LinearRegression

    meg_picks = mne.pick_types(raw.info, ref_meg=False, meg=True)
    ref_picks = mne.pick_types(raw.info, ref_meg=True, meg=False)
    if len(ref_picks) == 0:
        raise ValueError('Could not find meg ref channels.')

    estimator = LinearRegression(normalize=True)  # ref MAG + GRAD
    Y_pred = estimator.fit(
        raw[ref_picks][0][:, ::decim_fit].T,
        raw[meg_picks][0][:, ::decim_fit].T).predict(
        raw[ref_picks][0].T)
    raw._data[meg_picks] -= Y_pred.T


def map_ch_coords_to_mne(inst):
    """Transform sensors to MNE coordinates

    .. note::
        operates in place

    .. warning::
        For several reasons we do not use the MNE coordinates for the inverse
        modeling. This however won't always play nicely with visualization.

    Parameters
    ----------
    inst :  MNE data containers
        Raw, Epochs, Evoked.
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


def interpolate_missing(inst, subject, data_type, hcp_path,
                        run_index=0, mode='fast'):
    """Interpolate all MEG channels that are missing

    .. warning::
       This function may require some memory.

    Parameters
    ----------
    inst :  MNE data containers
        Raw, Epochs, Evoked.
    subject : str, file_map
        The subject
    data_type : str
        The kind of data to read. The following options are supported:
        'rest'
        'task_motor'
        'task_story_math'
        'task_working_memory'
        'noise_empty_room'
        'noise_subject'
    run_index : int
        The run index. For the first run, use 0, for the second, use 1.
        Also see HCP documentation for the number of runs for a given data
        type.
    hcp_path : str
        The HCP directory, defaults to op.curdir.
    mode : str
        Either `'accurate'` or `'fast'`, determines the quality of the
        Legendre polynomial expansion used for interpolation of MEG
        channels.

    Returns
    -------
    out :   MNE data containers
        Raw, Epochs, Evoked but with missing channels interpolated.
    """
    try:
        info = read_info(
            subject=subject, data_type=data_type, hcp_path=hcp_path,
            run_index=run_index if run_index is None else run_index)
    except (ValueError, IOError):
        raise ValueError(
            'could not find config to complete info.'
            'reading only channel positions without '
            'transforms.')

    # full BTI MEG channels
    bti_meg_channel_names = ['A%i' % ii for ii in range(1, 249, 1)]
    # figure out which channels are missing
    bti_meg_channel_missing_names = [
        ch for ch in bti_meg_channel_names if ch not in inst.ch_names]

    # get meg picks
    picks_meg = mne.pick_types(inst.info, meg=True, ref_meg=False)
    # some non-contiguous block in the middle so let's try to invert
    picks_other = [ii for ii in range(len(inst.ch_names)) if ii not in
                   picks_meg]
    other_chans = [inst.ch_names[po] for po in picks_other]

    # compute new n channels
    n_channels = (len(picks_meg) +
                  len(bti_meg_channel_missing_names) +
                  len(other_chans))

    # restrict info to final channels
    # ! info read from config file is not sorted like inst.info
    # ! therefore picking order matters, but we don't know it.
    # ! so far we will rely on the consistent layout for raw files
    final_names = [ch for ch in _data_labels if ch in bti_meg_channel_names or
                   ch in other_chans]
    info = _hcp_pick_info(info, final_names)
    assert len(info['ch_names']) == n_channels
    existing_channels_index = [ii for ii, ch in enumerate(info['ch_names']) if
                               ch in inst.ch_names]

    info['sfreq'] = inst.info['sfreq']

    # compute shape of data to be added
    is_raw = isinstance(inst, (mne.io.Raw,
                               mne.io.RawArray,
                               mne.io.bti.bti.RawBTi))
    is_epochs = isinstance(inst, (mne.Epochs, mne.EpochsArray))
    is_evoked = isinstance(inst, (mne.Evoked, mne.EvokedArray))
    if is_raw:
        shape = (n_channels,
                 (inst.last_samp - inst.first_samp) + 1)
        data = inst._data
    elif is_epochs:
        shape = (n_channels, len(inst.events), len(inst.times))
        data = np.transpose(inst.get_data(), (1, 0, 2))
    elif is_evoked:
        shape = (n_channels, len(inst.times))
        data = inst.data
    else:
        raise ValueError('instance must be Raw, Epochs '
                         'or Evoked')
    out_data = np.empty(shape, dtype=data.dtype)
    out_data[existing_channels_index] = data

    if is_raw:
        out = mne.io.RawArray(out_data, info)
        if inst.annotations is not None:
            out.annotations = inst.annotations
    elif is_epochs:
        out = mne.EpochsArray(data=np.transpose(out_data, (1, 0, 2)),
                              info=info, events=inst.events,
                              tmin=inst.times.min(), event_id=inst.event_id)
    elif is_evoked:
        out = mne.EvokedArray(
            data=out_data, info=info, tmin=inst.times.min(),
            comment=inst.comment, nave=inst.nave, kind=inst.kind)
    else:
        raise ValueError('instance must be Raw, Epochs '
                         'or Evoked')

    # set "bad" channels and interpolate.
    out.info['bads'] = bti_meg_channel_missing_names
    out.interpolate_bads(mode=mode)
    return out
