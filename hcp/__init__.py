# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

from . import viz
from . import preprocessing
from . import anatomy
from .anatomy import make_mne_anatomy, compute_forward_stack
from . import tests

from .io import read_raw
from .io import read_epochs
from .io import read_evokeds
from .io import read_info
from .io import read_annot
from .io import read_ica
from .io import read_trial_info

from .io import file_mapping

__version__ = '0.1.dev12'
