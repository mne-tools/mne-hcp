import os.path as op

import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_equal, assert_true, assert_raises


import mne
import hcp

hcp_path = op.join(op.dirname(op.dirname(op.dirname(__file__))),
                   'data', 'HCP')

_bti_chans = {'A' + str(i) for i in range(1, 249, 1)}

rest_subject = '100307'
task_subject = '104012'
task_types = ['task_story_math', 'task_working_memory', 'task_motor']
noise_types = ['noise_empty_room']
sfreq_preproc = 508.63
sfreq_raw = 2034.5101
lowpass_preproc = 150
highpass_preproc = 1.3


def test_read_annot():

    for run_index in range(3):
        annots = hcp.io.read_annot_hcp(subject=rest_subject, data_type='rest',
                                       hcp_path=hcp_path,
                                       run_index=run_index)
        # channels
        assert_equal(list(sorted(annots['channels'])),
                     ['all', 'ica', 'manual',  'neigh_corr',
                      'neigh_stdratio'])
        for channels in annots['channels'].values():
            for chan in channels:
                assert_true(chan in _bti_chans)

        # segments
        assert_equal(list(sorted(annots['ica'])),
                     ['bad', 'brain_ic', 'brain_ic_number',
                      'brain_ic_vs', 'brain_ic_vs_number',
                      'ecg_eog_ic', 'flag', 'good',
                      'physio', 'total_ic_number'])
        for components in annots['ica'].values():
            if len(components) > 0:
                assert_true(min(components) >= 0)
                assert_true(max(components) <= 248)


def _basic_raw_checks(raw):
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    assert_equal(len(picks), 248)
    ch_names = [raw.ch_names[pp] for pp in picks]
    assert_true(all(ch.startswith('A') for ch in ch_names))
    ch_sorted = list(sorted(ch_names))
    assert_true(ch_sorted != ch_names)
    assert_equal(np.round(raw.info['sfreq'], 4),
                 sfreq_raw)


def test_read_raw_rest():
    """Test reading raw for resting state"""
    for run_index in [0, 1, 2]:
        raw = hcp.io.read_raw_hcp(
            subject=rest_subject, hcp_path=hcp_path,
            data_type='rest', run_index=run_index)
        _basic_raw_checks(raw=raw)


def test_read_raw_task():
    """Test reading raw for tasks"""
    for run_index in [0, 1, 2]:
        for data_type in task_types:
            if run_index == 2:
                assert_raises(
                    ValueError, hcp.io.read_raw_hcp,
                    subject=task_subject, hcp_path=hcp_path,
                    data_type=data_type, run_index=run_index)
                continue
            raw = hcp.io.read_raw_hcp(
                subject=task_subject, hcp_path=hcp_path,
                data_type=data_type, run_index=run_index)
            _basic_raw_checks(raw=raw)


def test_read_raw_noise():
    """Test reading raw for empty room noise"""
    for run_index in [0, 1]:
        for data_type in noise_types:
            if run_index == 1:
                assert_raises(
                    ValueError, hcp.io.read_raw_hcp,
                    subject=task_subject, hcp_path=hcp_path,
                    data_type=data_type, run_index=run_index)
                continue
            raw = hcp.io.read_raw_hcp(
                subject=task_subject, hcp_path=hcp_path,
                data_type=data_type, run_index=run_index)
            _basic_raw_checks(raw=raw)


def _epochs_basic_checks(epochs, annots, data_type):
    n_good = 248 - len(annots['channels']['all'])
    if data_type == 'task_motor':
        n_good += 4
    assert_equal(len(epochs.ch_names), n_good)
    assert_equal(
        np.round(epochs.info['sfreq'], 2),
        sfreq_preproc)
    assert_array_equal(
        np.unique(epochs.events[:, 2]),
        np.array([99], dtype=np.int))


def test_read_epochs_rest():
    """Test reading epochs for resting state"""
    for run_index in [0]:
        annots = hcp.io.read_annot_hcp(
            subject=rest_subject, data_type='rest',
            hcp_path=hcp_path,
            run_index=run_index)

        epochs = hcp.io.read_epochs_hcp(
            subject=rest_subject, hcp_path=hcp_path,
            data_type='rest', run_index=run_index)

        _epochs_basic_checks(epochs, annots, data_type='rest')


def test_read_epochs_task():
    """Test reading epochs for taks"""
    for run_index in [0]:
        for data_type in task_types:
            annots = hcp.io.read_annot_hcp(
                subject=task_subject, data_type=data_type,
                hcp_path=hcp_path,
                run_index=run_index)

            epochs = hcp.io.read_epochs_hcp(
                subject=task_subject, hcp_path=hcp_path,
                data_type=data_type, run_index=run_index)

            _epochs_basic_checks(epochs, annots, data_type)


def test_read_evoked():
    pass


def test_read_ica():
    pass


def test_read_info():
    pass


def test_read_trial_info():
    pass
