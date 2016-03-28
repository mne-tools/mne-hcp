from .file_mapping import get_file_paths


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
                           processing='preprocessed',
                           mode=mode,
                           hcp_path=hcp_path_bucket))
    for output in meg_anatomy_outputs:
        aws_keys.extend(
            get_file_paths(subject=subject, data_type='meg_anatomy',
                           output=output,
                           processing='preprocessed',
                           hcp_path=hcp_path_bucket))
    return aws_keys


def get_s3_keys_meg(
        subject, data_types,
        processing=('preprocessed', 'unprocessed'),
        outputs=('meg_data', 'bads', 'ica'),
        run_inds=(0,), hcp_path_bucket='HCP_900'):
    """Helper to prepare AWS downloads """

    aws_keys = list()
    fun = get_file_paths
    for data_type in data_types:
        for output in outputs:
            for run_index in run_inds:
                if 'preprocessed' in processing and 'noise' not in data_type:
                    aws_keys.extend(
                        fun(subject=subject, data_type=data_type,
                            output=output, processing='preprocessed',
                            run_index=run_index,
                            hcp_path=hcp_path_bucket))
                if 'unprocessed' in processing and output == 'meg_data':
                    aws_keys.extend(
                        fun(subject=subject, data_type=data_type,
                            output=output, processing='unprocessed',
                            run_index=(run_index if 'noise' not in data_type
                                       else 0),
                            hcp_path=hcp_path_bucket))

    return aws_keys
