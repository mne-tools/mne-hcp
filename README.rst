
|Travis|_

.. |Travis| image:: https://api.travis-ci.org/mne-tools/mne-hcp.png?branch=master
.. _Travis: https://travis-ci.org/mne-tools/mne-hcp


MNE-HCP
=======

Python tools for processing the Human Connectome Date using MNE-Python.
Visit the `MNE-HCP homepage <http://mne-tools.github.io/mne-hcp>`_ for full user documentation.

Scope and Disclaimer
--------------------

This code is under active research-driven development
and the API is still changing but is getting closer to a stable release.

.. note::

    For now please consider the following caveats:

    - we only intend to support a subset of the files shipped with HCP.
    - Precisely, for now it is not planned to support io and processing for any outputs of the HCP source space pipelines.
    - This library breaks with some of MNE conventions in order to make the HCP outputs compatible with MNE.

Installation
============

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. To install ``mne-hcp``, you first need to install its dependencies::

	$ pip install numpy matplotlib scipy scikit-learn mne joblib pandas

Then clone the repository::

	$ git clone http://github.com/mne-tools/mne-hcp

and finally run `setup.py` to install the package::

	$ cd mne-hcp/
	$ python setup.py install

If you do not have admin privileges on the computer, use the ``--user`` flag
with `setup.py`.

For a devoloper install based on symbolic links do::

	$ cd mne-hcp/
	$ python setup.py develop

To check if everything worked fine, you can do::

	$ python -c 'import hcp'

and it should not give any error messages.

Dependencies
------------

The following main and additional dependencies are required to enjoy MNE-HCP:
- MNE-Python master branch
- scipy
- numpy
- matplotlib
- scikit-learn (optional)

Quickstart
==========

The following data layout is expected. A folder that contains the HCP data
as they are unpacked by a zip, subject wise.
When data were downloaded via the Aspera connect client the following
command should produce the expected layout.

.. code-block:: bash

   $ for fname in $(ls *zip); do
   $    echo unpacking $fname;
   $    unzip -o $fname; rm $fname;
   $ done

When files are downloaded using the amazon webserviced tools, e.g. `s3rcmd`,
all should be fine.

The code is organized by different modules.
The `io` module includes readers for sensor space data at different processing
stages and annotations for baddata.

Reader API
----------

All data readers have the same API for the first two positional arguments:


.. code-block:: python

   params = dict(
       subject='1003007',
       data_type='task_motor')  # assuming that data are unpacked here

   # all MNE objects have native names and coordinates, some MNE functions might break.
   info = hcp.io.read_info_hcp(**params)  # MNE object
   raw = hcp.io.read_raw_hcp(**params)  # ...
   epochs = hcp.io.read_epochs_hcp(**params) # ...
   list_of_evoked = hcp.io.read_evokeds_hcp(**params) # ...
   annotations_dict = hcp.io.read_annot_hcp(**params) # dict
   ica_mat = hcp.io.read_ica_hcp(**params) # ...

Types of Data
-------------

MNE-HCP uses custom names for values that are more mne-hcpic, the following
table gives an overview

+-----------------------+-------------------------------------+----------------+
| **name**              | **readers**                         | **HCP jargon** |
+-----------------------+-------------------------------------+----------------+
| 'rest'                | raw, epochs, info, annotations, ica | 'Restin''      |
+-----------------------+-------------------------------------+----------------+
| 'task_working_memory' | raw, epochs, info, annotations, ica | 'Wrkmem'       |
+-----------------------+-------------------------------------+----------------+
| 'task_story_math'     | raw, epochs, info, annotations, ica | 'StoryM'       |
+-----------------------+-------------------------------------+----------------+
| 'task_motor'          | raw, epochs, info, annotations, ica | 'Motor'        |
+-----------------------+-------------------------------------+----------------+
| 'noise_subject'       | raw, info                           | 'Pnoise'       |
+-----------------------+-------------------------------------+----------------+
| 'noise_empty_room'    | raw, info                           | 'Rnoise'       |
+-----------------------+-------------------------------------+----------------+

Functionality to make the HCP compatible with MNE
-------------------------------------------------

MNE HCP comes with convenience functions such as `hcp.make_mne_anatomy`. This one willcreate an
MNE friendly anatomy directories and extractes the head model and
coregistration MEG to MRI coregistration. Yes it maps to MRI, not to the
helmet -- a peculiarity of the HCP data.
It can be used as follows:

.. code-block:: python

   >>> import os.path as op
   >>> import hcp
   >>> storage_dir = op.expanduser('~/data/MNE-HCP')
   >>>  hcp.make_mne_anatomy(
   >>>      '100307', anatomy_path=storage_dir + '/subjects',
   >>>      hcp_path=storage_dir + '/HCP',
   >>>      recordings_path=storage_dir + '/hcp-meg')
   reading extended structural processing ...
   reading RAS freesurfer transform
   Combining RAS transform and coregistration
   extracting head model
   coregistring head model to MNE-HCP coordinates
   extracting coregistration


File Mapping
------------

MNE-HCP supports a low level file mapping that allows for quick compilations
of sets of files for a given subejct and data context.
This is done in :func:`hcp.io.file_mapping.get_file_paths`, think of it as a
file name synthesizer that takes certain data description parameters as inputs
and lists all corresponding files.

Example usage:

.. code-block:: python

   >>> import hcp
   >>> files = hcp.io.file_mapping.get_file_paths(
   >>>     subject='123455', data_type='task_motor', output='raw',
   >>>     hcp_path='/media/crazy_disk/HCP')
   ['/media/crazy_disk/HCP/123455/unprocessed/MEG/10-Motor/4D/c,rfDC',
    '/media/crazy_disk/HCP/123455/unprocessed/MEG/10-Motor/4D/config']

Why we are not globbing files? Because the HCP-MEG data are fixed, all file
patterns are known and access via Amazon web services easier if the files
to be accessed are known in advance.

Gotchas
=======

Native coordinates and resulting plotting and processing peculartities
----------------------------------------------------------------------

The HCP for MEG provides coregistration information for native BTI/4D
setting. MNE-Python expects coordinates in meters and the Neuromag
right anterior superior (RAS) coordinates. However, essential information is
missing to compute all transforms needed to easily perform the conversions.

For now, the way things work, all processing is performed in native BTI/4D
coordinates with the device-to-head transform skipped (set to identity matrix),
such that the coregistration directly maps from the native 4D sensors,
represented in head coordinates, to the freesurfer space. This has a few minor
consequences that you may confusing to MNE-Python users.

1. In the reader code you will see many flags set to ```convert=False```, etc.
This is not a bug.

2. All channel names and positions are native, topographic plotting might not
work as as expected. First of all the layout file is not recognized, second,
the coordinates are not regonized as native ones, eventually rotating and
distorting the graphical display. To fix this either a proper layout can be
computed with :func:`hcp.viz.make_hcp_bti_layout`.
The conversion to MNE can be
performed too using :func:`hcp.preprocessing.map_ch_coords_to_mne`.
But note that source localization will be wrong when computerd on data in
Neuromag coordinates. As things are coordinates have to be kept in the native
space to be aligned with the HCP outputs.

Reproducing HCP sensor space outputs
------------------------------------

A couple of steps are necessary to reproduce
the original sensor space outputs.

1. Reference channels should be regressed out. Checkout :func:`hcp.preprocessing.apply_ref_correction`.

2. The trial info structure gives the correct latencies of the events
   The latencies in the trigger channel are shifted by around 18 ms.
   For now we'd recommend using the events from the :func:`hcp.io.read_trial_info_hcp`.

3. The default filters in MNE and FieldTrip are different.
   FieldTrip uses 4th order butterworth filter. In MNE you might need
   to adjust the `*_trans_bandwidth` parameter to avoid numerical error.
   In the HCP outputs evoked responses were filtered between 0.5 and 30Hz prior
   to baseline correction.

4. Annotations need to be loaded and registered. The HCP consortium ships annotations of bad segments and bad channels.
   These have to be read and used. Checkout `hcp.io.read_annot_hcp` and add bad
   channel neame to `raw.info['bads']` and create and set an mne.Annotations
   object as atribute to raw, see below.

    .. code-block:: python

        annots = hcp.io.read_annot_hcp(subject, data_type, hcp_path=hcp_path,
                                       run_index=run_index)
        bad_segments = annots['segments']['all'] / raw.info['sfreq']
        raw.annotations = mne.Annotations(
            bad_segments[:, 0], (bad_segments[:, 1] - bad_segments[:, 0]),
            description='bad')

5. ICA components related to eye blinks and heart beats need to be removed
   from the data. Checkout the ICA slot in the output of
   `hcp.io.read_annot_hcp` to get the HCP ICA components.


Convenience functions
---------------------

NNE-HCP ships convenience functions that help set up directory and file layouts
expected by MNE-Python.

:func:`hcp.make_mne_anatomy` will produce an MNE and Freesurfer compatible directory layout and will create the following outputs by default, mostly using sympbolic links:

.. code-block:: bash
    $anatomy_path/$subject/bem/inner_skull.surf
    $anatomy_path/$subject/label/*
    $anatomy_path/$subject/mri/*
    $anatomy_path/$subject/surf/*
    $recordings_path/$subject/$subject-head_mri-trans.fif

These can then be set as $SUBJECTS_DIR and as MEG directory, consistent
with MNE examples.
Here, `inner_skull.surf` and `$subject-head_mri-trans.fif` are written  by the function such that they can be used by MNE. The latter is the coregistration matrix.

Python Indexing
^^^^^^^^^^^^^^^

MNE-HCP corrects on reading the indices it finds for data segments, events, or
components. The indices it reads from the files will already be mapped to
Python convention by subtracring 1.

Contributions
-------------
- currently `@dengemann` is pushing frequently to master, if you plan to contribute, open issues and pull requests, or contact `@dengemann` directly. Discussions are welcomed.

Acknowledgements
================

This project is supported by the AWS Cloud Credits fo Research program and
bu the ERC starting grant ERC StG 263584 issued to Virginie van Wassenhove.

I acknowledge support by Alex Gramfort, Mainak Jas, Jona Sassenhagen, Giorgos Michalareas,
Eric Larson, Danilo Bzdok, and Jan-Mathijs Schoffelen for discussions,
inputs and help with finding the best way to map
HCP data to the MNE world.
