from .file_mapping import get_file_paths, run_map


def get_s3_keys_anatomy(
        subject,
        freesurfer_outputs=('label', 'mri', 'surf'),
        meg_anatomy_outputs=('head_model', 'transforms'),
        hcp_path_bucket='HCP_900'):
    """Helper to prepare AWS downloads for anatomy data

    A helper function useful for working with Amazon EC2 and S3.
    It compiles a list of related files.

    .. note::
        This function does not download anything.
        It only facilitates downloading by compiling the content.

    Parameters
    ----------
    subject : str
        The subject, a 6 digit code.
    freesurfer_outputs : str | list | tuple
        The Freesurfer outputs to be downloaded. Defaults to
        `('label', 'mri', 'surf')`.
    meg_anatomy_outputs : str | list | tuple
        The MEG anatomy contents to download. Defaults to
        `('head_model', 'transforms')`.
    hcp_path_bucket : str
        The S3 bucket path. Will be prepended to each file path.

    Returns
    -------
    aws_keys : list of str
        The AWS S3 keys to be downloaded.
    """
    aws_keys = list()
    for output in freesurfer_outputs:
        aws_keys.extend(
            get_file_paths(subject=subject, data_type='freesurfer',
                           output=output,
                           hcp_path=hcp_path_bucket))
    for output in meg_anatomy_outputs:
        aws_keys.extend(
            get_file_paths(subject=subject, data_type='meg_anatomy',
                           output=output,
                           hcp_path=hcp_path_bucket))
    return aws_keys


def get_s3_keys_meg(
        subject, data_types, outputs=('raw', 'bads', 'ica'),
        run_inds=0, hcp_path_bucket='HCP_900', onsets='stim'):
    """Helper to prepare AWS downloads for MEG data

    A helper function useful for working with Amazon EC2 and S3.
    It compiles a list of related files.

    .. note::
        This function does not download anything.
        It only facilitates downloading by compiling the content.

    Parameters
    ----------
    subject : str
        The subject, a 6 digit code.
    data_type : str | tuple of str | list of str
        The acquisition context of the data. The following ones are supported:
        'rest'
        'noise'
        'task_motor'
        'task_story_math'
        'task_working_memory'
    outputs : str | tuple of str | list of str
        The kind of output. The following ones are supported:
        'raw'
        'epochs'
        'evoked'
        'ica'
        'annotations'
        'trial_info'
        'freesurfer'
        'meg_anatomy'
    onsets : {'stim', 'resp', 'sentence', 'block'} | list | tuple
        The event onsets. Only considered for epochs and evoked outputs
        The mapping is generous, everything that is not a response is a
        stimulus, in the sense of internal or external events. sentence and
        block are specific to task_story_math. Can be a collection of those.
    run_inds : int | list of int | tuple of int
        The run index. For the first run, use 0, for the second, use 1.
        Also see HCP documentation for the number of runs for a given data
        type.
    hcp_path_bucket : str
        The S3 bucket path. Will be prepended to each file path.

    Returns
    -------
    aws_keys : list of str
        The AWS S3 keys to be downloaded.
    """
    aws_keys = list()
    fun = get_file_paths
    if not isinstance(onsets, (list, tuple)):
        onsets = [onsets]
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if not isinstance(run_inds, (list, tuple)):
        run_inds = [run_inds]
    if not isinstance(data_types, (list, tuple)):
        data_types = [data_types]

    if not all(isinstance(rr, int) for rr in run_inds):
        raise ValueError('Rund indices must be integers. I found: ' +
                         ', '.join(['%s' % type(rr) for rr in run_inds
                                    if not isinstance(rr, int)]))
    elif max(run_inds) > 2:
        raise ValueError('For HCP MEG data there is no task with more'
                         'than three runs. Among your requests there '
                         'is run index %i. Did you forget about '
                         'zero-based indexing?' % max(run_inds))
    elif min(run_inds) < 0:
        raise ValueError('Run indices must be positive')

    for data_type in data_types:
        for output in outputs:
            if 'noise' in data_type and output != 'raw':
                continue  # we only have raw for noise data
            elif data_type == 'rest' and output in ('evoked', 'trial_info'):
                continue  # there is no such thing as evoked resting state data
            for run_index in run_inds:
                if run_index + 1 > len(run_map[data_type]):
                    continue  # ignore irrelevant run indices
                for onset in onsets:
                    aws_keys.extend(
                        fun(subject=subject, data_type=data_type,
                            output=output, run_index=run_index, onset=onset,
                            hcp_path=hcp_path_bucket))

    return aws_keys
