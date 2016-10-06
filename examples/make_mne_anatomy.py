"""
.. _tut_make_anatomy:

=====================================
Extract essential anatomy information
=====================================

The HCP anatomy data needed for MNE based source analysis
Live in different places and subfolders.
MNE-HCP has a convenience function that exctracts the required
files and information and creates a "subjects_dir" as known from
MNE and freesurfer.

The most essential contents are
- freesurfer surfaces
- the MRI
- the headmodel
- the coregistration transform

This function is your friend prior to running any source analysis.
"""
# Author: Denis A. Engemann
# License: BSD 3 clause

import os.path as op
import hcp
storage_dir = op.expanduser('~/mne-hcp-data')

# This is where the data are after downloading from HCP
hcp_path = storage_dir + '/HCP'

# this is the new directory to be created
subjects_dir = storage_dir + '/hcp-subjects'

# it will make the subfolders 'bem', 'mir', 'surf' and 'label'
# and populate it with symbolic links if possible and write some files
# where information has to be presented in a different format to satisfy MNE.

# this is where the coregistration matrix is written as
# `105923-head_mri-transform.fif`
recordings_path = storage_dir + '/hcp-meg'

# comment out to write files on your disk
# hcp.make_mne_anatomy(
#     subject='105923', hcp_path=hcp_path,
#     recordings_path=recordings_path,
#     subjects_dir=subjects_dir)
