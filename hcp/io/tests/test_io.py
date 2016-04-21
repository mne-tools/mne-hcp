import os.path as op

from nose.tools import assert_equal, assert_true

import hcp

hcp_path = op.join(op.dirname(op.dirname(__file__)), 'data', 'HCP')

_bti_chans = {'A' + str(i) for i in range(1, 249, 1)}


def test_read_annot():

    for run_index in range(3):
        annots = hcp.io.read_annot_hcp(subject='100307', data_type='rest',
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
