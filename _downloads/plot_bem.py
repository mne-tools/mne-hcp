"""
============================================================
Plot the head model inside MRI to check anatomical alignment
============================================================

MNE-HCP facilitates source level analysis by supporting linkage between
MRT/anatomy files and the MEG data.
Here, inspect the head model that is used to compute the BEM solution.
"""
# Author: Jona Sassenhagen
#         Denis A. Engemann
# License: BSD 3 clause

import os.path as op

from mne.viz import plot_bem
# from mne.bem import make_watershed_bem

storage_dir = op.expanduser('~/mne-hcp-data')
subject = '105923'
subjects_dir = storage_dir + '/hcp-subjects'

###############################################################################
# We assume that all directory structures are set up.
# Checkout :func:`hcp.make_mne_anatomy` to achieve this necessary step.
# See also :ref:`tut_make_anatomy`.

plot_bem(subject=subject, subjects_dir=subjects_dir)
