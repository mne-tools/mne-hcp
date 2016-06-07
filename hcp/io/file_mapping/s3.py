from .file_mapping import get_file_paths, run_map


def get_s3_keys_anatomy(
        subject,
        freesurfer_outputs=('label', 'mri', 'surf'),
        meg_anatomy_outputs=('head_model', 'transforms'),
        mode='minimal',
        hcp_path_bucket='HCP_900'):
    """Helper to prepare AWS downloads """
    aws_keys = list()
    for output in freesurfer_outputs:
        aws_keys.extend(
            get_file_paths(subject=subject, data_type='freesurfer',
                           output=output,
                           mode=mode,
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
    """Helper to prepare AWS downloads """
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
            elif data_type == 'rest' and output == 'evoked':
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
