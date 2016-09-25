# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

from . import config

import os
import os.path as op
from subprocess import call


def _download_testing_data():
    """download testing data
    .. note::
        requires python 2.7
    """
    for s3key in config.s3_keys:
        new_path = op.dirname(s3key).split(config.hcp_prefix)[-1][1:]
        new_path = op.join(config.hcp_path, new_path)
        fname = op.basename(s3key)
        new_file = op.join(new_path, fname)
        if not op.exists(new_path):
            os.makedirs(new_path)
        if not op.exists(new_file):
            print('downloading:\n\tfrom %s\n\tto %s' % (s3key, new_path))
            call(['s3cmd', 'get', s3key, new_path], shell=False)
            assert op.exists(new_file)
