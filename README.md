# MNE-HCP

Python tools for processing HCP data using MNE-Python

## disclaimer and goals

This code is under active, research-driven development and the API is still unstable. At a later stage this code will likely be wrapped by MNE-Python to provide a more common API. For now consider the following caveats:
- we only intend to support a subset of the files shipped with HCP. Precisely, for now it is not planned to support io and processing for any outputs of the HCP inverse pipelines.
- the code is not covered by unit tests so far as I did not have the time to create mock testing data.
- this library breaks with some of MNE conventions due to peculiarities of the HCP data shipping policy. The basic IO is based on paths, not on files.
The core API is a file map, a list of dictionaries that map all available files
that we have for a given subject. This is needed because the information required to construct MNE objects is scattered across several files.

## contributions
- currently `@dengemann` is pushing frequently to master, if you plan to contribute, open issues and pull requests, or contact `@dengemann` directly. Discussions are welcomed.

## dependencies
- MNE-Python master branch
- scipy
- numpy
- matplotlib

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

```Python
hcp.io.read_info_hcp  # get channel info for rest | tasks and a given run
hcp.io.read_raw_hcp  # same for raw data
hcp.io.read_epochs_hcp  # same for epochs epochs
hcp.io.read_ica_hcp  # ica solution as dict
hcp.io.read_annot_hcp  # bad channels, segments and ICA annotations
```

### reader API

All data readers have the same non-default API:

```Python
params = dict(
    subject='1003007',
    data_type='task_motor')  # assuming that data are unpacked here

# all MNE objects have native names and coordinates, some MNE functions might
# break.
info = hcp.io.read_info_hcp(**params)  # MNE object
raw = hcp.io.read_raw_hcp(**params)  # ...
epochs = hcp.io.read_epochs_hcp(**params) # ...
annotations_dict = hcp.io.read_annot_hcp(**params) # dict
ica_dict = hcp.io.read_ica_hcp(**params) # ...

```

### data kinds

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

### workflows: scripts in a function to map HCP to MNE worlds 

For convenience several workflows are provieded. Currently the most supported
is `hcp.workflows.make_mne_anatomy`, a convenience function that will create
MNE friendly anatomy directories and extractes the head model and
coregistration MEG to MRI coregistration. Yes it maps to MRI, not to the
helmet -- a peculiarity of the HCP data.
It can be used as follows:

```Python
hcp.workflows.anatomy.make_mne_anatomy(
    subject='100307', hcp_path='/media/crazy_disk/HCP',
    anatomy_path='/home/crazy_user/hcp-subjects',
    recordings_path='/home/crazy_user/hcp-meg',
    mode='full') # consider "minimal" for linking and writing less 
```

# acknowledgements 

This project is supported by the AWS Cloud Credits for Research program.
