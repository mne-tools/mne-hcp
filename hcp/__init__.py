# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

from . import anatomy, preprocessing, viz
from ._version import __version__
from .anatomy import compute_forward_stack, make_mne_anatomy
from .io import (
    file_mapping,
    read_annot,
    read_epochs,
    read_evokeds,
    read_ica,
    read_info,
    read_raw,
    read_trial_info,
)
