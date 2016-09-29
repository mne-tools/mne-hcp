"""
============================
Read and plot coregistration
============================

We'll take a look at the coregistration here.
"""
# Author: Denis A. Enegemann
# License: BSD 3 clause

import os.path as op

from hcp.viz import plot_coregistration

##############################################################################
# we assume our data is inside a designated folder under $HOME
storage_dir = op.expanduser('~/mne-hcp-data')

##############################################################################
# and we assume to have the downloaded data, the MNE/freesurfer style
# anatomy directory, and the MNE style MEG directory.
# these can be obtained from :func:`make_mne_anatomy`.
# See also :ref:`tut_make_anatomy`.

hcp_params = dict(
    subject='105923',
    hcp_path=op.join(storage_dir, 'HCP'),
    subjects_dir=op.join(storage_dir, 'subjects'),
    recordings_path=op.join(storage_dir, 'hcp-meg'))

##############################################################################
# let's plot two views

for azim in (0, 90):
    plot_coregistration(
        view_init=(('azim', azim), ('elev', 0)), **hcp_params)
