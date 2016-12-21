# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import os
import os.path as op

hcp_prefix = 's3://hcp-openaccess/HCP_900'
subject = '105923'


hcp_data_types = [
    'rest',
    'task_working_memory',
    'task_story_math',
    'task_motor',
    'noise_empty_room'
]


hcp_outputs = [
    'raw',
    'epochs',
    'ica',
    'evoked',
    'trial_info',
    'bads'
]

hcp_onsets = ['stim']

# allow for downloading fewer data
run_inds = [0, 1, 2]
max_runs = int(os.getenv('MNE_HCP_N_RUNS', 3))

##############################################################################
# variable used in different tests

hcp_path = op.expanduser('~/mne-hcp-data/mne-hcp-testing')

bti_chans = {'A' + str(i) for i in range(1, 249, 1)}

test_subject = '105923'
task_types = ['task_story_math', 'task_working_memory', 'task_motor']
noise_types = ['noise_empty_room']
test_decim = 100
sfreq_preproc = 508.63 / test_decim
sfreq_raw = 2034.5101
lowpass_preproc = 150
highpass_preproc = 1.3

epochs_bounds = {
    'task_motor': (-1.2, 1.2),
    'task_working_memory': (-1.5, 2.5),
    'task_story_math': (-1.5, 4),
    'rest': (0, 2)
}


def nottest(f):
    """Decorator to mark a function as not a test"""
    f.__test__ = False
    return f


@nottest
def expensive_test(f):
    """Decorator for expensive testing"""
    f.expensive_test = True
    return f
