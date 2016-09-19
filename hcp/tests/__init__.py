# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

from . import config

import os
import os.path as op
from subprocess import call, PIPE


def _download_testing_data():
    """ download testing data
    .. note::
        requires python 2.7
    """
    for s3key in config.s3_keys:
        new_path = op.dirname(s3key).split(config.hcp_prefix)[-1][1:]
        new_path = op.join(config.hcp_path, 'HCP', new_path)
        if not op.exists(new_path):
            os.makedirs(new_path)
        print('downloading:\n\tfrom %s\n\tto %s' % (s3key, new_path))
        call(['s3cmd', 'get', s3key, new_path], shell=True, stdout=PIPE)
