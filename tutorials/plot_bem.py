"""
==================================
Compute forward model and plot BEM
==================================

MNE-HCP facilitates source level analysis by supporting linkage between MRT/anatomy files and the MEG data.
Here, we prepare the anatomy files and show the BEM.
"""
# Author: Jona Sassenhagen
# License: BSD 3 clause

import os.path as op

import hcp
from mne import plot_bem
from mne.bem import make_watershed_bem

hcp_path = op.expanduser('~/mne-hcp-data')
subject = '100307'
subjects_dir = hcp_path + 'subjects'

########################################################
# A simple way for creating all of the files required by
# the MNE-Python source localisation pipelines is provided
# by the function hcp.make_mne_anatomy function, e.g.
# hcp.make_mne_anatomy(**anatomy_params)
# But here we create the BEM by hand.

mne.bem.make_watershed_bem(subject, subjects_dir=subjects_dir, overwrite=False, verbose=None)

########################################################
# Now we can plot the BEM solution. We are using a single-layer BEM because HCP only has MEG data.

mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir)
