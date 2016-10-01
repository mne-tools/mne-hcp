"""
==========================================================
Interpolate channel that are missing in an HCP output file
==========================================================

We'll take a look at the coregistration here.
"""
# Author: Denis A. Enegemann
# License: BSD 3 clause

import os.path as op

import mne

import hcp
from hcp import preprocessing as preproc

mne.set_log_level('WARNING')

##############################################################################
# we assume our data is inside a designated folder under $HOME
# and set the IO params

storage_dir = op.expanduser('~')

hcp_params = dict(hcp_path=op.join(storage_dir, 'mne-hcp-data', 'HCP'),
                  subject='105923',
                  data_type='task_working_memory')

##############################################################################
# we take the some evoked and create an interpolated copy

evoked = hcp.read_evokeds(**hcp_params)[0]

# The HCP pipelines don't interpolate missing channels
print('%i channels out of 248 expected' % len(evoked.ch_names))

evoked_interpolated = preproc.interpolate_missing(evoked, **hcp_params)

##############################################################################
# Let's visualize what has changed!

# we calculate the difference ...
bads = set(evoked_interpolated.ch_names) - set(evoked.ch_names)
print(bads)

# ... and mark the respective channels as bad ...
evoked_interpolated.info['bads'] += list(bads)

# ... such that MNE is displaying the interpolated time series in red ...
evoked_interpolated.plot(exclude=[])
