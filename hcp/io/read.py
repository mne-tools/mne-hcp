# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import itertools as itt
import os.path as op
import re

import numpy as np
import scipy.io as scio
from scipy import linalg

from mne import (EpochsArray, EvokedArray, pick_info,
                 rename_channels)
from mne.io.bti.bti import _get_bti_info, read_raw_bti
from mne.io import _loc_to_coil_trans
from mne.utils import logger

from .file_mapping import get_file_paths

_data_labels = [
    'TRIGGER',
    'RESPONSE',
    'MLzA',
    'MLyA',
    'MLzaA',
    'MLyaA',
    'MLxA',
    'A22',
    'MLxaA',
    'A2',
    'MRzA',
    'MRxA',
    'MRzaA',
    'MRxaA',
    'MRyA',
    'MCzA',
    'MRyaA',
    'MCzaA',
    'MCyA',
    'GzxA',
    'MCyaA',
    'A104',
    'SA1',
    'A241',
    'MCxA',
    'A138',
    'MCxaA',
    'A214',
    'SA2',
    'SA3',
    'A71',
    'A26',
    'A93',
    'A39',
    'A125',
    'A20',
    'A65',
    'A9',
    'A8',
    'A95',
    'A114',
    'A175',
    'A16',
    'A228',
    'A35',
    'A191',
    'A37',
    'A170',
    'A207',
    'A112',
    'A224',
    'A82',
    'A238',
    'A202',
    'A220',
    'A28',
    'A239',
    'A13',
    'A165',
    'A204',
    'A233',
    'A98',
    'A25',
    'A70',
    'A72',
    'A11',
    'A47',
    'A160',
    'A64',
    'A3',
    'A177',
    'A63',
    'A155',
    'A10',
    'A127',
    'A67',
    'A115',
    'A247',
    'A174',
    'A194',
    'A5',
    'A242',
    'A176',
    'A78',
    'A168',
    'A31',
    'A223',
    'A245',
    'A219',
    'A12',
    'A186',
    'A105',
    'A222',
    'A76',
    'A50',
    'A188',
    'A231',
    'A45',
    'A180',
    'A99',
    'A234',
    'A215',
    'A235',
    'A181',
    'A38',
    'A230',
    'A91',
    'A212',
    'A24',
    'A66',
    'A42',
    'A96',
    'A57',
    'A86',
    'A56',
    'A116',
    'A151',
    'A141',
    'A120',
    'A189',
    'A80',
    'A210',
    'A143',
    'A113',
    'A27',
    'A137',
    'A135',
    'A167',
    'A75',
    'A240',
    'A206',
    'A107',
    'A130',
    'A100',
    'A43',
    'A200',
    'A102',
    'A132',
    'A183',
    'A199',
    'A122',
    'A19',
    'A62',
    'A21',
    'A229',
    'A84',
    'A213',
    'A55',
    'A32',
    'A85',
    'A146',
    'A58',
    'A60',
    'GyyA',
    'A88',
    'A79',
    'GzyA',
    'GxxA',
    'A169',
    'A54',
    'GyxA',
    'A203',
    'A145',
    'A103',
    'A163',
    'A139',
    'A49',
    'A166',
    'A156',
    'A128',
    'A68',
    'A159',
    'A236',
    'A161',
    'A121',
    'A4',
    'A61',
    'A6',
    'A126',
    'A14',
    'A94',
    'A15',
    'A193',
    'A150',
    'A227',
    'A59',
    'A36',
    'A225',
    'A195',
    'A30',
    'A109',
    'A172',
    'A108',
    'A81',
    'A171',
    'A218',
    'A173',
    'A201',
    'A74',
    'A29',
    'A164',
    'A205',
    'A232',
    'A69',
    'A157',
    'A97',
    'A217',
    'A101',
    'A124',
    'A40',
    'A123',
    'A153',
    'A178',
    'A1',
    'A179',
    'A33',
    'A147',
    'A117',
    'A148',
    'A87',
    'A89',
    'A243',
    'A119',
    'A52',
    'A142',
    'A211',
    'A190',
    'A53',
    'A192',
    'A73',
    'A226',
    'A136',
    'A184',
    'A51',
    'A237',
    'A77',
    'A129',
    'A131',
    'A198',
    'A197',
    'A182',
    'A46',
    'A92',
    'A41',
    'A90',
    'A7',
    'A23',
    'A83',
    'A154',
    'A34',
    'A17',
    'A18',
    'A248',
    'A149',
    'A118',
    'A208',
    'A152',
    'A140',
    'A144',
    'A209',
    'A110',
    'A111',
    'A244',
    'A185',
    'A246',
    'A162',
    'A106',
    'A187',
    'A48',
    'A221',
    'A196',
    'A133',
    'A158',
    'A44',
    'A134',
    'A216',
    'UACurrent',
    'ECG+',
    'VEOG+',
    'HEOG+',
    'EMG_LF',
    'EMG_LH',
    'ECG-',
    'VEOG-',
    'HEOG-',
    'EMG_RF',
    'EMG_RH'
]

_label_mapping = [
    ('E1', 'ECG+'),
    ('E3', 'VEOG+'),
    ('E5', 'HEOG+'),
    ('E63', 'EMG_LF'),
    ('E31', 'EMG_LH'),
    ('E2', 'ECG-'),
    ('E4', 'VEOG-'),
    ('E6', 'HEOG-'),
    ('E64', 'EMG_RF'),
    ('E32', 'EMG_RH')
]

_time_lock_mapping = dict(
    TRESP='resp',
    TEMG='resp',
    TIM='stim',
    TEV='stim',
    TFLA='stim',
    BSENT='stim',
    BU='stim'
)


def _parse_trans(string):
    """helper to parse transforms"""
    return np.array(string.replace('\n', '')
                          .strip('[] ')
                          .split(' '), dtype=float).reshape(4, 4)


def _parse_hcp_trans(fid, transforms, convert_to_meter):
    """"another helper"""
    contents = fid.read()
    for trans in contents.split(';'):
        if 'filename' in trans or trans == '\n':
            continue
        key, trans = trans.split(' = ')
        key = key.lstrip('\ntransform.')
        transforms[key] = _parse_trans(trans)
        if convert_to_meter:
            transforms[key][:3, 3] *= 1e-3  # mm to m
    if not transforms:
        raise RuntimeError('Could not parse the transforms.')


def _read_trans_hcp(fname, convert_to_meter):
    """Read + parse transforms
    subject_MEG_anatomy_transform.txt
    """
    transforms = dict()
    with open(fname) as fid:
        _parse_hcp_trans(fid, transforms, convert_to_meter)
    return transforms


def _read_landmarks_hcp(fname):
    """XXX parse landmarks currently not used"""
    out = dict()
    with open(fname) as fid:
        for line in fid:
            kind, data = line.split(' = ')
            kind = kind.split('.')[1]
            if kind == 'coordsys':
                out['coord_frame'] = data.split(';')[0].replace("'", "")
            else:
                data = data.split()
                for c in ('[', '];'):
                    if c in data:
                        data.remove(c)
                out[kind] = np.array(data, dtype=int) * 1e-3  # mm to m
    return out


def _get_head_model(head_model_fname):
    """helper to parse head model from matfile"""
    head_mat = scio.loadmat(head_model_fname, squeeze_me=False)
    pnts = head_mat['headmodel']['bnd'][0][0][0][0][0]
    faces = head_mat['headmodel']['bnd'][0][0][0][0][1]
    faces -= 1  # correct for Matlab's 1-based index
    return pnts, faces


def _read_bti_info(raw_fid, config):
    """helper to only access bti info from pdf file"""
    info, bti_info = _get_bti_info(
        pdf_fname=raw_fid, config_fname=config, head_shape_fname=None,
        rotation_x=0.0, translation=(0.0, 0.02, 0.11),
        ecg_ch='E31', eog_ch=('E63', 'E64'),
        convert=False,  # no conversion to neuromag coordinates
        rename_channels=False,  # keep native channel names
        sort_by_ch_name=False)  # do not change native order
    return info


def _read_raw_bti(raw_fid, config_fid, convert, verbose=None):
    """Convert and raw file from HCP input"""
    raw = read_raw_bti(  # no convrt + no rename for HCP compatibility
        raw_fid, config_fid, convert=convert, head_shape_fname=None,
        sort_by_ch_name=False, rename_channels=False, preload=False,
        verbose=verbose)

    return raw


def _check_raw_config_runs(raws, configs):
    """XXX this goes to tests later, currently not used """
    for raw, config in zip(raws, configs):
        assert op.split(raw)[0] == op.split(config)[0]
    run_str = set([configs[0].split('/')[-3]])
    for config in configs[1:]:
        assert set(configs[0].split('/')) - set(config.split('/')) == run_str


def _check_infos_trans(infos):
    """XXX this goes to tests later, currently not used"""
    chan_max_idx = np.argmax([c['nchan'] for c in infos])
    chan_template = infos[chan_max_idx]['ch_names']
    channels = [c['ch_names'] for c in infos]
    common_channels = set(chan_template).intersection(*channels)

    common_chs = [[c['chs'][c['ch_names'].index(ch)] for ch in common_channels]
                  for c in infos]
    dev_ctf_trans = [i['dev_ctf_t']['trans'] for i in infos]
    cns = [[c['ch_name'] for c in cc] for cc in common_chs]
    for cn1, cn2 in itt.combinations(cns, 2):
        assert cn1 == cn2
    # BTI stores data in head coords, as a consequence the coordinates
    # change across run, we apply the ctf->ctf_head transform here
    # to check that all transforms are correct.
    cts = [np.array([linalg.inv(_loc_to_coil_trans(c['loc'])).dot(t)
                    for c in cc])
           for t, cc in zip(dev_ctf_trans, common_chs)]
    for ct1, ct2 in itt.combinations(cts, 2):
        np.testing.assert_array_almost_equal(ct1, ct2, 12)


def read_raw(subject, data_type, run_index=0, hcp_path=op.curdir,
             verbose=None):
    """Read HCP raw data

    Parameters
    ----------
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    raw : instance of mne.io.Raw
        The MNE raw object.
    """
    pdf, config = get_file_paths(
        subject=subject, data_type=data_type, output='raw',
        run_index=run_index, hcp_path=hcp_path)

    raw = _read_raw_bti(pdf, config, convert=False, verbose=verbose)
    return raw


def read_info(subject, data_type, run_index=0, hcp_path=op.curdir):
    """Read info from unprocessed data

    Parameters
    ----------
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

    Returns
    -------
    info : instance of mne.io.meas_info.Info
        The MNE channel info object.

    .. note::
        HCP MEG does not deliver only 3 of the 5 task packages from MRI HCP.
    """
    raw, config = get_file_paths(
        subject=subject, data_type=data_type, output='raw',
        run_index=run_index, hcp_path=hcp_path)

    if not op.exists(raw):
        raw = None

    meg_info = _read_bti_info(raw, config)

    if raw is None:
        logger.info('Did not find Raw data. Guessing EMG, ECG and EOG '
                    'channels')
        rename_channels(meg_info, dict(_label_mapping))
    return meg_info


def read_epochs(subject, data_type, onset='stim', run_index=0,
                    hcp_path=op.curdir, return_fixations_motor=False):
    """Read HCP processed data

    Parameters
    ----------
    subject : str, file_map
        The subject
    data_type : str
        The kind of data to read. The following options are supported:
        'rest'
        'task_motor'
        'task_story_math'
        'task_working_memory'
    onset : {'stim', 'resp', 'sentence', 'block'}
        The event onset. The mapping is generous, everything that is not a
        response is a stimulus, in the sense of internal or external events.
        `sentence` and `block` are specific to task_story_math.
    run_index : int
        The run index. For the first run, use 0, for the second, use 1.
        Also see HCP documentation for the number of runs for a given data
        type.
    hcp_path : str
        The HCP directory, defaults to op.curdir.
    return_fixations_motor : bool
        Weather to return fixations or regular trials. For motor data only.
        Defaults to False.
    Returns
    -------
    epochs : instance of mne.Epochs
        The MNE epochs. Note, these are pseudo-epochs in the case of
        onset == 'rest'.
    """
    info = read_info(subject=subject, data_type=data_type,
                         run_index=run_index, hcp_path=hcp_path)

    epochs_mat_fname = get_file_paths(
        subject=subject, data_type=data_type, output='epochs',
        onset=onset,
        run_index=run_index, hcp_path=hcp_path)[0]

    if data_type != 'task_motor':
        return_fixations_motor = None
    epochs = _read_epochs(epochs_mat_fname=epochs_mat_fname, info=info,
                          return_fixations_motor=return_fixations_motor)
    if data_type == 'task_motor':
        epochs.set_channel_types(
            {ch: 'emg' for ch in epochs.ch_names if 'EMG' in ch})
    return epochs


def _read_epochs(epochs_mat_fname, info, return_fixations_motor):
    """read the epochs from matfile"""
    data = scio.loadmat(epochs_mat_fname,
                        squeeze_me=True)['data']
    ch_names = [ch for ch in data['label'].tolist()]
    info['sfreq'] = data['fsample'].tolist()
    times = data['time'].tolist()[0]
    # deal with different event lengths
    if return_fixations_motor is not None:
        fixation_mask = data['trialinfo'].tolist()[:, 1] == 6
        if return_fixations_motor is False:
            fixation_mask = ~fixation_mask
        data = np.array(data['trial'].tolist()[fixation_mask].tolist())
    else:
        data = np.array(data['trial'].tolist().tolist())

    # warning: data are not chronologically ordered but
    # match the trial info
    events = np.zeros((len(data), 3), dtype=np.int)
    events[:, 0] = np.arange(len(data))
    events[:, 2] = 99  # all events
    # we leave it to the user to construct his events
    # as from the data['trialinfo'] arbitrary events can be constructed.
    # and it is task specific.
    this_info = _hcp_pick_info(info, ch_names)
    epochs = EpochsArray(data=data, info=this_info, events=events,
                         tmin=times.min())
    # XXX hack for now due to issue with EpochsArray constructor
    # cf https://github.com/mne-tools/mne-hcp/issues/9
    epochs.times = times
    return epochs


def _hcp_pick_info(info, ch_names):
    """helper to subset info"""
    return pick_info(
        info, [info['ch_names'].index(ch) for ch in ch_names],
        copy=True)


def read_trial_info(subject, data_type, run_index=0, hcp_path=op.curdir):
    """Read information about trials

    Parameters
    ----------
    subject : str
        The HCP subject.
    data_type : str
        The kind of data to read. The following options are supported:
        'rest'
        'task_motor'
        'task_story_math'
        'task_working_memory'
    run_index : int
        The run index. For the first run, use 0, for the second, use 1.
        Also see HCP documentation for the number of runs for a given data
        type.
    hcp_path : str
        The HCP directory, defaults to op.curdir.
    Returns
    -------
    trial_info : dict
        The trial info including event labels, indices and times.
    """

    trial_info_mat_fname = get_file_paths(
        subject=subject, data_type=data_type,
        output='trial_info', run_index=run_index,
        hcp_path=hcp_path)[0]

    trl_info = _read_trial_info(trial_info_mat_fname=trial_info_mat_fname)
    return trl_info


def _read_trial_info(trial_info_mat_fname):
    """helper to read trial info"""
    # XXX FIXME index -1
    data = scio.loadmat(trial_info_mat_fname, squeeze_me=True)['trlInfo']
    out = dict()

    for idx, lock_name in enumerate(data['lockNames'].tolist()):
        key = _time_lock_mapping[lock_name]
        out[key] = dict(
            comments=data['trlColDescr'].tolist()[idx],
            codes=data['lockTrl'].tolist().tolist()[idx])

    return out


def _check_sorting_runs(candidates, id_char):
    """helper to ensure correct run-parsing and mapping"""
    run_idx = [f.find(id_char) for f in candidates]
    for config, idx in zip(candidates, run_idx):
        assert config[idx - 1].isdigit()
        assert not config[idx - 2].isdigit()
    runs = [int(f[idx - 1]) for f, idx in zip(candidates, run_idx)]
    return runs, candidates


def _parse_annotations_segments(segment_strings):
    """Read bad segments defintions from text file"""
    for char in '}]':  # multi line array definitions
        segment_strings = segment_strings.replace(
            char + ';', 'splitme'
        )
    split = segment_strings.split('splitme')
    out = dict()
    for entry in split:
        if len(entry) == 1 or entry == '\n':
            continue
        key, rest = entry.split(' = ')
        val = np.array(
            [k for k in [''.join([c for c in e if c.isdigit()])
             for e in rest.split()] if k.isdigit()], dtype=int)
        # reshape and map to Python index
        val = val.reshape(-1, 2) - 1
        out[key.split('.')[1]] = val
    return out


def read_annot(subject, data_type, run_index=0, hcp_path=op.curdir):
    """Read annotations for bad data and ICA.

    Parameters
    ----------
    subject : str, file_map
        The subject
    data_type : str
        The kind of data to read. The following options are supported:
        'rest'
        'task_motor'
        'task_story_math'
        'task_working_memory'
    run_index : int
        The run index. For the first run, use 0, for the second, use 1.
        Also see HCP documentation for the number of runs for a given data
        type.
    hcp_path : str
        The HCP directory, defaults to op.curdir.

    Returns
    -------
    out : dict
        The annotations.
    """
    bads_files = get_file_paths(
        subject=subject, data_type=data_type,
        output='bads', run_index=run_index,
        hcp_path=hcp_path)
    segments_fname = [k for k in bads_files if
                      k.endswith('baddata_badsegments.txt')][0]
    bads_fname = [k for k in bads_files if
                  k.endswith('baddata_badchannels.txt')][0]

    ica_files = get_file_paths(
        subject=subject, data_type=data_type,
        output='ica', run_index=run_index,
        hcp_path=hcp_path)
    ica_fname = [k for k in ica_files if k.endswith('icaclass_vs.txt')][0]

    out = dict()
    iter_fun = [
        ('channels', _parse_annotations_bad_channels, bads_fname),
        ('segments', _parse_annotations_segments, segments_fname),
        ('ica', _parse_annotations_ica, ica_fname)]

    for subtype, fun, fname in iter_fun:
        with open(fname, 'r') as fid:
            out[subtype] = fun(fid.read())

    return out


def read_ica(subject, data_type, run_index=0, hcp_path=op.curdir):
    """Read precomputed independent components from subject

    Parameters
    ----------
    subject : str, file_map
        The subject
    data_type : str
        The kind of data to read. The following options are supported:
        'rest'
        'task_motor'
        'task_story_math'
        'task_working_memory'
    run_index : int
        The run index. For the first run, use 0, for the second, use 1.
        Also see HCP documentation for the number of runs for a given data
        type.
    hcp_path : str
        The HCP directory, defaults to op.curdir.

    Returns
    -------
    mat : numpy structured array
        The ICA mat struct.
    """

    ica_files = get_file_paths(
        subject=subject, data_type=data_type,
        output='ica', run_index=run_index,
        hcp_path=hcp_path)
    ica_fname_mat = [k for k in ica_files if k.endswith('icaclass.mat')][0]

    mat = scio.loadmat(ica_fname_mat, squeeze_me=True)['comp_class']
    return mat


def _parse_annotations_bad_channels(bads_strings):
    """Read bad channel definitions from text file"""
    for char in '}]':
        bads_strings = bads_strings.replace(
            char + ';', 'splitme'
        )
    split = bads_strings.split('splitme')
    out = dict()
    for entry in split:
        if len(entry) == 1 or entry == '\n':
            continue
        key, rest = entry.split(' = ')
        val = [ch for ch in rest.split("'") if ch.isalnum()]
        out[key.split('.')[1]] = val
    return out


def _parse_annotations_ica(ica_strings):
    """Read bad channel definitions from text file"""
    # prepare splitting
    for char in '}]':  # multi line array definitions
        ica_strings = ica_strings.replace(
            char + ';', 'splitme'
        )
    # scalar variables
    match_inds = list()
    for match in re.finditer(';', ica_strings):
        ii = match.start()
        if ica_strings[ii - 1].isalnum():
            match_inds.append(ii)

    ica_strings = list(ica_strings)
    for ii in match_inds:
        ica_strings[ii] = 'splitme'
    ica_strings = ''.join(ica_strings)

    split = ica_strings.split('splitme')
    out = dict()
    for entry in split:
        if len(entry) == 1 or entry == '\n':
            continue
        key, rest = entry.split(' = ')
        if '[' in rest:
            sep = ' '
        else:
            sep = "'"
        val = [ch for ch in rest.split(sep) if ch.isalnum()]
        if all(v.isdigit() for v in val):
            val = [int(v) - 1 for v in val]  # map to Python index
        out[key.split('.')[1]] = val
    return out


def read_evokeds(subject, data_type, onset='stim', sensor_mode='mag',
                     hcp_path=op.curdir, kind='average'):
    """Read HCP processed data

    Parameters
    ----------
    subject : str, file_map
        The subject
    data_type : str
        The kind of data to read. The following options are supported:
        'rest'
        'task_motor'
        'task_story_math'
        'task_working_memory'
    onset : {'stim', 'resp'}
        The event onset. The mapping is generous, everything that is not a
        response is a stimulus, in the sense of internal or external events.
    sensor_mode : {'mag', 'planar'}
        The sensor projection. Defaults to 'mag'. Only relevant for
        evoked output.
    hcp_path : str
        The HCP directory, defaults to op.curdir.
    kind : {'average', 'standard_error'}
        The averaging mode. Defaults to 'average'.
    Returns
    -------
    epochs : instance of mne.Epochs
        The MNE epochs. Note, these are pseudo-epochs in the case of
        onset == 'rest'.
    """
    info = read_info(subject=subject, data_type=data_type,
                         hcp_path=hcp_path, run_index=0)

    evoked_files = list()
    for fname in get_file_paths(
            subject=subject, data_type=data_type, onset=onset,
            output='evoked', sensor_mode=sensor_mode, hcp_path=hcp_path):
        evoked_files.extend(_read_evoked(fname, sensor_mode, info, kind))
    return evoked_files


def _read_evoked(fname, sensor_mode, info, kind):
    """helper to read evokeds"""
    data = scio.loadmat(fname, squeeze_me=True)['data']
    ch_names = [ch for ch in data['label'].tolist()]

    times = data['time'].tolist()
    sfreq = 1. / np.diff(times)[0]

    info = _hcp_pick_info(info, ch_names)
    info['sfreq'] = sfreq

    out = list()
    comment = ('_'.join(fname.split('/')[-1].split('_')[2:])
                  .replace('.mat', '')
                  .replace('_eravg_', '_')
                  .replace('[', '')
                  .replace(']', ''))
    nave = np.unique(data['dof'].tolist())
    assert len(nave) == 1
    nave = nave[0]
    for key, this_kind in (('var', 'standard_error'), ('avg', 'average')):
        if this_kind != kind:
            continue
        evoked = EvokedArray(
            data=data[key].tolist(), info=info, tmin=min(times),
            kind=this_kind, comment=comment, nave=nave)
        out.append(evoked)
    return out
