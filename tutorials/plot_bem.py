"""
==================================
Compute forward model and plot BEM
==================================

MNE-HCP makes it easy to construct inverse models (for source localisation) from HCP data.
Here, we compute the full "forward stack" and graphically check if the BEM solution is well-behaved.
"""
# Author: Jona Sassenhagen
# License: BSD 3 clause

import hcp
import mne

hcp_path = '/Volumes/backup/'  # XXX: fix
subject = '100307'
subjects_dir = hcp_path + 'subjects'

anatomy_params = dict(
    subject=subject, anatomy_path=subjects_dir,
    hcp_path=hcp_path + 'HCP',
    recordings_path=hcp_path + 'hcp-meg'
    )

########################################################
# First, create the required anatomical files

hcp.make_mne_anatomy(**anatomy_params)

########################################################
# Next, compute the 'forward stack':
# With one handy function, construct everything required for inverse modelling.
# Results are stored in a dict.
# Note this step can take up to a few minutes;
# particularly calculating the forward model is computationally expensive.

forward_results = hcp.compute_forward_stack(**anatomy_params)
print(forward_results.keys())

########################################################
# Now we can plot the BEM solution.
# We are using a single-layer BEM because HCP only has MEG data.

bem = out["bem_sol"]
mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir)
