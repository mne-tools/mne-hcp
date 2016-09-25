#! /usr/bin/env python
#

# Copyright (C) 2015-2016 Denis Engemann
# <denis.engemann@gmail.com>

import os
from os import path as op

import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(os.path.join('hcp', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


descr = """MNE HCP project for accessing the human connectome MEG data in Python.""" # noqa

DISTNAME = 'mne-hcp'
DESCRIPTION = descr
MAINTAINER = 'Denis A. Engemann'
MAINTAINER_EMAIL = 'denis.engemann@gmail.com'
URL = 'http://github.com/mne-tools/mne-hcp'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'http://github.com/mne-tools/mne-hcp'
VERSION = version


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.rst').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=['hcp',
                    'hcp.io',
                    'hcp.io.tests',
                    'hcp.io.file_mapping',
                    'hcp.io.file_mapping.tests',
                    'hcp.tests'],
          package_data={'hcp': [
              op.join('io', 'file_mapping', 'data', '*txt')]})
