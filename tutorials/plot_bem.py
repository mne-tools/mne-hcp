"""
==================================
Compute forward model and plot BEM
==================================

MNE-HCP makes it easy to construct inverse models (for source localisation) from HCP data.
Here, we compute the full "forward stack" and graphically check if the BEM solution is well-behaved.
"""
# Author: Jona Sassenhagen
# License: BSD 3 clause

import os.path as op

import hcp
from mne import plot_bem

hcp_path = op.expanduser('~/mne-hcp-data')
subject = '100307'
subjects_dir = hcp_path + 'subjects'

anatomy_params = dict(
    subject=subject, anatomy_path=subjects_dir,
    hcp_path=hcp_path + 'HCP',
    recordings_path=hcp_path + 'hcp-meg'
    )

########################################################
# This function assumes that we have created the required anatomical files.
# A simple way for doing that is using the function hcp.make_mne_anatomy;
# it takes similar parameters as the hcp.compute_forward_stack function
# we will be using next, so you could just run hcp.make_mne_anatomy(**anatomy_params)
# to set up the anatomy files.

# But for now, we first, compute the 'forward stack':
# With one handy function, construct everything required for inverse modelling.
# Results are stored in a dict.
# Note this step can take up to a few minutes;
# particularly calculating the forward model/morphing is computationally expensive.

forward_results = hcp.compute_forward_stack(**anatomy_params)
print(forward_results.keys())

########################################################
# We can plot the BEM solution. We are using a single-layer BEM because HCP only has MEG data.

mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir)

########################################################
# We can also plot the sensitivity map based on the forward model.
# (This 3D plot requires the Python2 packge Mayavi to be installed.)

fwd = forward_results["fwd"]
grad_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')
grad_map.plot(time_label='Sensitivity', subjects_dir=subjects_dir + 'T1w/',
              clim=dict(lims=[0, 50, 100]))
