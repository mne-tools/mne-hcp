"""
.. _tut_plot_evoked:

=====================
Visualize evoked data
=====================

Here we'll generate some attention for pecularities of visualizing the
HCP evoked outputs using MNE plotting functions.
"""
# Author: Denis A. Enegemann
# License: BSD 3 clause

import os.path as op
import hcp
from hcp import preprocessing as preproc

##############################################################################
# we assume our data is inside a designated folder under $HOME
storage_dir = op.expanduser('~/mne-hcp-data')
hcp_path = op.join(storage_dir, 'HCP')
recordings_path = op.join(storage_dir, 'hcp-meg')
subjects_dir = op.join(storage_dir, 'hcp-subjects')
subject = '105923'  # our test subject
data_type = 'task_working_memory'
run_index = 0


##############################################################################
# Let's get the evoked data.

hcp_evokeds = hcp.read_evokeds(onset='stim', subject=subject,
                               data_type=data_type, hcp_path=hcp_path)
for evoked in hcp_evokeds:
    if not evoked.comment == 'Wrkmem_LM-TIM-face_BT-diff_MODE-mag':
        continue
##############################################################################
# In order to plot topographic patterns we need to transform the sensor
# positions to MNE coordinates one a copy of the data.
# We're not using this these transformed data for any source analyses.
# These take place in native BTI/4D coordinates.

evoked_viz = evoked.copy()
preproc.map_ch_coords_to_mne(evoked_viz)
evoked_viz.plot_joint()

##############################################################################
# for subsequent analyses we would use `evoked` not `evoked_viz`.
# See also :ref:`tut_compute_inverse_erf` to see how the story continues.
