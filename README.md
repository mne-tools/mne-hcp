# MNE-HCP

Python tools for processing HCP data using MNE-Python

## disclaimer and goals

This code is under active, research-driven development
and the API is still unstable.
At a later stage this code will likely be wrapped by MNE-Python to provide a
more common API. For now consider the following caveats:
- we only intend to support a subset of the files shipped with HCP. Precisely, for now it is not planned to support io and processing for any outputs of the
- HCP inverse pipelines.
- the code is not covered by unit tests so far as I did not have the time to create mock testing data.
- this library breaks with some of MNE conventions due to peculiarities of the HCP data shipping policy. The basic IO is based on paths, not on files.

## dependencies

The following main and additional dependencies are required to enjoy MNE-HCP:
- MNE-Python master branch
- scipy
- numpy
- matplotlib
- scikit-learn (additional)

## usage

The following data layout is expected. A folder that contains the HCP data
as they are unpacked by a zip, subject wise. See command that will produce this
layout.

```bash
for fname in $(ls *zip); do
    echo unpacking $fname;
    unzip -o $fname; rm $fname;
done
```

The code is organized by different modules.
The `io` module includes readers for sensor space data at different processing
stages and annotations for baddata.

These are (all native coordinates + names):

```python
hcp.io.read_info_hcp  # get channel info for rest | tasks and a given run
hcp.io.read_raw_hcp  # same for raw data
hcp.io.read_epochs_hcp  # same for epochs epochs
hcp.io.read_ica_hcp  # ica solution as dict
hcp.io.read_annot_hcp  # bad channels, segments and ICA annotations
```

### reader API

All data readers have the same API for the first two positional arguments:

```python
params = dict(
    subject='1003007',
    data_type='task_motor')  # assuming that data are unpacked here

# all MNE objects have native names and coordinates, some MNE functions might
# break.
info = hcp.io.read_info_hcp(**params)  # MNE object
raw = hcp.io.read_raw_hcp(**params)  # ...
epochs = hcp.io.read_epochs_hcp(**params) # ...
list_of_evoked = hcp.io.read_evokeds_hcp(**params) # ...
annotations_dict = hcp.io.read_annot_hcp(**params) # dict
ica_dict = hcp.io.read_ica_hcp(**params) # ...
```

### types of data

MNE-HCP uses custom names for values that are more mne-pythonic, the following
table gives an overview

| name                  | readers                             | HCP jargon |
|-----------------------|-------------------------------------|------------|
| 'rest'                | raw, epochs, info, annotations, ica | 'Restin''  |
| 'task_working_memory' | raw, epochs, info, annotations, ica | 'Wrkmem'   |
| 'task_story_math'     | raw, epochs, info, annotations, ica | 'StoryM'   |
| 'task_motor'          | raw, epochs, info, annotations, ica | 'Motor'    |
| 'noise_subject'       | raw, info                           | 'Pnoise'   |
| 'noise_empty_room'    | raw, info                           | 'Rnoise'   |


### workflows scripts in a function to map HCP to MNE worlds

For convenience several workflows are provieded. Currently the most supported
is `hcp.workflows.make_mne_anatomy`, a convenience function that will create
MNE friendly anatomy directories and extractes the head model and
coregistration MEG to MRI coregistration. Yes it maps to MRI, not to the
helmet -- a peculiarity of the HCP data.
It can be used as follows:

```python
hcp.workflows.anatomy.make_mne_anatomy(
    subject='100307', hcp_path='/media/crazy_disk/HCP',
    anatomy_path='/home/crazy_user/hcp-subjects',
    recordings_path='/home/crazy_user/hcp-meg',
    mode='full') # consider "minimal" for linking and writing less
```

### low level file mapping

One core element of MNE-HCP is a file mapping that allows for quick selections
of files for a given subejct and data context.
This is done in `hcp.io.file_mapping.get_file_paths`, think of it as a
file name synthesizer that takes certain data description parameters as inputs
and lists all corresponding files.

Example usage:

```Python
files = hcp.io.file_mapping.get_file_paths(
    subject='123455', data_type='task_motor', output='raw',
    hcp_path='/media/crazy_disk/HCP')

print(files)
# output:
['/media/crazy_disk/HCP/123455/unprocessed/MEG/10-Motor/4D/c,rfDC',
 '/media/crazy_disk/HCP/123455/unprocessed/MEG/10-Motor/4D/config']
```

Why we are not globbing files? Because the HCP-MEG data are fixed, all file
patterns are known and access via Amazon web services easier if the files
to be accessed are known in advance.

## Gotchas

### Native coordinates and resulting plotting and processing peculartities

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
computed with ```hcp.preprocessing.make_hcp_bti_layout```.
The conversion to MNE can be
performed too using ```hcp.preprocessing.transform_sensors_to_mne```.
But note that source localization will be wrong when computerd on data in
Neuromag coordinates. As things are coordinates have to be kept in the native
space to be aligned with the HCP outputs.

### Reproducing HCP sensor space outputs

A couple of steps are necessary to reproduce
the original sensor space outputs.

#### Reference channels should be regressed out.

Checkout `hcp.preprocessing.apply_ref_meg_residual_fit`.

#### The trial info structure gives the correct latencies of the events

The latencies in the trigger channel are shifted by around 18 ms.
For now we'd recommend using the events from the `hcp.io.read_trial_info_hcp`.

#### The default filters in MNE and FieldTrip are different.

FieldTrip uses 4th order butterworth filter. In MNE you might need
to adjust the `*_trans_bandwidth` parameter to avoid numerical error.
In the HCP outputs evoked responses were filtered between 0.5 and 30Hz prior
to baseline correction.

#### Annotations need to be loaded and registered

The HCP consortium ships annotations of bad segments and bad channels.
These have to be read and used. Checkout `hcp.io.read_annot_hcp` and add bad
channel neame to `raw.info['bads']` and create and set an mne.Annotations
object as atribute to raw, see below.

```Python
annots = hcp.io.read_annot_hcp(subject, data_type, hcp_path=hcp_path,
                               run_index=run_index)
bad_segments = annots['segments']['all']
raw.annotations = mne.Annotations(
    bad_segments[:, 0], (bad_segments[:, 1] - bad_segments[:, 0]),
    description='bad')
```

#### ICA components

ICA components related to eye blinks and heart beats need to be removed
from the data. Checkout the ICA slot in the output of
`hcp.io.read_annot_hcp` to get the HCP ICA components.


### Workflows

NNE-HCP ships convenience functions, called workflows to perform stereotypical
tasks that are required for using standard MNE code.

#### anatomy

`hcp.workflows.anatomy.make_mne_anatomy` will produce an MNE and Freesurfer compatible directory layout and will create the following outputs by default, mostly using sympbolic links:

```bash
$anatomy_path/$subject/bem/inner_skull.surf
$anatomy_path/$subject/label/*
$anatomy_path/$subject/mri/*
$anatomy_path/$subject/surf/*
$recordings_path/$subject/$subject-head_mri-trans.fif
```

These can then be set as $SUBJECTS_DIR and as MEG directory, consistent
with MNE examples.
Here, `inner_skull.surf` and `$subject-head_mri-trans.fif` are written  by the function such that they can be used by MNE. The latter is the coregistration matrix.

#### inverse

`hcp.workflows.inverse.make_mne_forward` computes the bem model, the source space for a given subject and for fsaverage.
It then computes the forward solution
in using the morphed source space
(from fsaverage to the subject).

### Python Indexing

MNE-HCP corrects on reading the indices it finds for data segments, events, or
components. The indices it reads from the files will already be mapped to
Python convention by subtracring 1.

## contributions
- currently `@dengemann` is pushing frequently to master, if you plan to contribute, open issues and pull requests, or contact `@dengemann` directly. Discussions are welcomed.

### Unit tests

Are in the making, however at this points you still need to download the HCP
data to run them.

# Acknowledgements

This project is supported by the AWS Cloud Credits for Research program.
Thanks Alex Gramfort, Giorgos Michalareas, Eric Larson and Jan-Mathijs
Schoffelen for discussions, inputs and help with finding the best way to map
HCP data to the MNE world. Thanks Virginie van Wassenhove for supporting this
project.
