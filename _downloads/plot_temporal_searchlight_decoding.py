# -*- coding: utf-8 -*-
"""
.. _tut_searchlight_decoding:

=======================================================
Run temporal searchlight decoding on event related data
=======================================================

In this tutorial we show how to run a temporal window decoding
on event related data. We'll try to decode tools VS faces in the
working memory data.
"""
# Author: Denis A. Enegemann
# License: BSD 3 clause

import os.path as op

import numpy as np
import mne
import hcp
from hcp import preprocessing as preproc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from mne.decoding import GeneralizationAcrossTime

mne.set_log_level('WARNING')

# we assume our data is inside its designated folder under $HOME
storage_dir = op.expanduser('~')
hcp_params = dict(
    hcp_path=op.join(storage_dir, 'mne-hcp-data', 'HCP'),
    subject='105923',
    data_type='task_working_memory')

# these values are looked up from the HCP manual
tmin, tmax = -1.5, 2.5
decim = 3

##############################################################################
# We know from studying either the manual or the trial info about the mapping
# of events.
event_id = dict(face=1, tool=2)

##############################################################################
# we first collect epochs across runs and essentially adopt the code
# shown in :ref:`tut_reproduce_erf`.

epochs = list()
for run_index in [0, 1]:
    hcp_params['run_index'] = run_index
    trial_info = hcp.read_trial_info(**hcp_params)

    events = np.c_[
        trial_info['stim']['codes'][:, 6] - 1,  # time sample
        np.zeros(len(trial_info['stim']['codes'])),
        trial_info['stim']['codes'][:, 3]  # event codes
    ].astype(int)

    # for some reason in the HCP data the time events may not always be unique
    unique_subset = np.nonzero(np.r_[1, np.diff(events[:, 0])])[0]
    events = events[unique_subset]  # use diff to find first unique events
    subset = np.in1d(events[:, 2], event_id.values())
    epochs_hcp = hcp.read_epochs(**hcp_params).decimate(decim)
    epochs_hcp = epochs_hcp[unique_subset][subset]
    epochs_hcp.events[:, 2] = events[subset, 2]
    epochs_hcp.event_id = event_id
    epochs_hcp.crop(-0.1, 0.5)
    epochs.append(preproc.interpolate_missing(epochs_hcp, **hcp_params))

epochs = mne.concatenate_epochs(epochs)
del epochs_hcp

##############################################################################
# Now we can proceed as shown in the MNE-Python decoding tutorials

y = LabelBinarizer().fit_transform(epochs.events[:, 2]).ravel()

cv = StratifiedKFold(y=y)  # do a stratified cross-validation

gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
                               cv=cv, scorer=roc_auc_score)
# fit and score
gat.fit(epochs, y=y)
gat.score(epochs)

##############################################################################
# Ploting the temporal connectome and the evolution of discriminability.
gat.plot()
gat.plot_diagonal()
