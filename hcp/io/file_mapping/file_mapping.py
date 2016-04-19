import os.path as op

""" Notes

For now:
- source pipeline are not considered
- datacheck pipelines are not considered
- EPRIME files are ignored

the following string formatters are used:

subject : the subject ID
run : the run ID
kind : the type of recording, e.g. 'Restin', 'Wrkmem'
pipeline : the name of the pipeline, e.g., 'icaclass'
context : either 'rmeg' or 'tmeg'
condition : the condition label for average pipelines
diff_modes : the contrast label for average pipelines, e.g. '[BT-diff]'
    or '[OP-diff]', etc.
sensor_mode : 'MODE-mag', 'MODE-grad'
"""

unprocessed = {
    'path': '{subject}/unprocessed/MEG/{run}-{kind}/4D',
    'patterns': ['c,rfDC', 'config'],
}


preprocessed = {
    'meg': {
        'path': '{subject}/MEG/{kind}/{pipeline}/',
        'patterns': {
            ('meg_data', 'rmeg'): [
                '{subject}_MEG_{run}-{kind}_{context}preproc.mat'],
            ('meg_data', 'tmeg'): [
                '{subject}_MEG_{run}-{kind}_tmegpreproc_TIM.mat',
                '{subject}_MEG_{run}-{kind}_tmegpreproc_TRESP.mat'
            ],
            'bads': [
                '{subject}_MEG_{run}-{kind}_baddata_badchannels.txt',
                '{subject}_MEG_{run}-{kind}_baddata_badsegments.txt',
                '{subject}_MEG_{run}-{kind}_baddata_manual_badchannels.txt',
                '{subject}_MEG_{run}-{kind}_baddata_manual_badsegments.txt'
            ],
            'ica': [
                '{subject}_MEG_{run}-{kind}_icaclass_vs.mat',
                '{subject}_MEG_{run}-{kind}_icaclass_vs.txt',
                '{subject}_MEG_{run}-{kind}_icaclass.mat',
                '{subject}_MEG_{run}-{kind}_icaclass.txt'
            ],
            'psd': ['{subject}_MEG_{run}-{kind}_powavg.mat'],
            'evoked': [
                ('{subject}_MEG_{kind}_eravg_[{condition}]_{diff_modes}_'
                 '[{sensor_mode}].mat')
                ],
            'tfr': [
                ('{subject}_MEG_{kind}_tfavg_[{condition}]_{diff_modes}_'
                 '[{sensor_mode}].mat')
            ],
            'trial_info': [
                '{subject}_MEG_{run}-{kind}_tmegpreproc_trialinfo.mat'
            ]
        }
    },
    'meg_anatomy': {
        'path': '{subject}/MEG/anatomy',
        'patterns': {
            'transforms': [
                '{subject}_MEG_anatomy_transform.txt',
            ],
            'head_model': [
                '{subject}_MEG_anatomy_headmodel.mat'
            ],
            'source_model': [
                '{subject}_MEG_anatomy_sourcemodel_2d.mat',
                '{subject}_MEG_anatomy_sourcemodel_3d4mm.mat',
                '{subject}_MEG_anatomy_sourcemodel_3d6mm.mat',
                '{subject}_MEG_anatomy_sourcemodel_3d8mm.mat'
            ],
            'freesurfer': [
                '{subject}.L.inflated.4k_fs_LR.surf.gii',
                '{subject}.R.inflated.4k_fs_LR.surf.gii',
                '{subject}.L.midthickness.4k_fs_LR.surf.gii']
        }
    },
    'freesurfer': {
        'path': '{subject}/T1w/{subject}',
        'patterns': {
            'label': [],
            'surf': [],
            'mri': [],
            'stats': [],
            'touch': []
        }
    }
}

file_modes = {
    'freesurfer':
        {'minimal': [
            'c_ras.mat',
            '001.mgz',
            'T1.mgz',
            'lh.white',
            'rh.white',
            'lh.inflated',
            'rh.inflated',
            'lh.sphere.reg',
            'rh.sphere.reg']},
}


freesurfer_files = op.join(op.dirname(__file__), 'data', '%s.txt')
for kind, patterns in preprocessed['freesurfer']['patterns'].items():
    with open(freesurfer_files % kind) as fid:
        patterns.extend([k.rstrip('\n') for k in fid.readlines()])

pipeline_map = {
    'ica': 'icaclass',
    'bads': 'baddata',
    'psd': 'powavg',
    'evoked': 'rravg',
    'tfr': 'tfavg'
}

kind_map = {
    'task_motor': 'Motor',
    'task_working_memory': 'Wrkmem',
    'task_story_math': 'StoryM',
    'rest': 'Restin',
    'noise_empty_room':  'Rnoise',
    'noise_subject': 'subject',
    'meg_anatomy': 'anatomy',
    'freesurfer': 'freesurfer'
}

run_map = {
    'noise_empty_room': ['1'],
    'noise_subject': ['2'],
    'rest': ['3', '4', '5'],
    'task_working_memory': ['6', '7'],
    'task_story_math': ['8', '9'],
    'task_motor': ['10', '11'],
    'meg_anatomy': [],
    'freesurfer': []
}

onset_map = {
    'stim': 'TIM',
    'resp': 'TRESP'
}


def get_file_paths(subject, data_type, output, processing, run_index=0,
                   onset='auto', conditions=(), diff_modes=(),
                   mode='full',
                   sensor_modes=(), hcp_path='.'):
    if data_type not in kind_map:
        raise ValueError('I never heard of `%s` -- are you sure this is a'
                         ' valid HCP type? I currenlty support:\n%s' % (
                             data_type, ' \n'.join(
                                 [k for k in kind_map if '_' in k])))

    context = ('rmeg' if 'rest' in data_type else 'tmeg')
    if onset == 'auto':
        if data_type == 'task_story_math':
            onset = 'resp'
        else:
            onset = 'stim'
    elif onset == 'stim' and data_type == 'task_story_math':
        raise ValueError('No stimulus locked data are available for %s' %
                         data_type)

    my_onset = onset_map[onset]
    if data_type not in ('meg_anatomy', 'freesurfer'):
        my_runs = run_map[data_type]
        if run_index >= len(my_runs):
            raise ValueError('For `data_type=%s` we have %d runs. '
                             'You asked for run index %d.' % (
                                 data_type, len(my_runs), run_index))
        run_label = my_runs[run_index]
    else:
        run_label = None
    files = list()
    output_key = (data_type if output == 'trial_info' else output)
    pipeline = pipeline_map.get(output_key, output_key)
    if processing == 'preprocessed':
        file_map = preprocessed[(data_type if data_type in (
                                 'meg_anatomy', 'freesurfer') else 'meg')]
        path = file_map['path'].format(
            subject=subject,
            pipeline=(context + 'preproc' if output == 'meg_data'
                      else pipeline),
            kind=kind_map[data_type])

        if output == 'meg_data':
            pattern_key = (output, context)
        else:
            pattern_key = output

        my_pattern = file_map['patterns'][pattern_key]
        if data_type == 'task_story_math':  # story math has only resp
            my_pattern = [pp for pp in my_pattern if 'TIM.mat' not in pp]

        if output in ('bads', 'ica'):
            files.extend(
                [op.join(path,
                         p.format(subject=subject, run=run_label,
                                  kind=kind_map[data_type]))
                 for p in my_pattern])

        elif output == 'meg_data':
            if 'noise' in data_type:
                raise ValueError('You want preprocessed data of type "%s". '
                                 'But the HCP does not ship those. Sorry.' %
                                 data_type)
            if context == 'rmeg':
                my_pattern = my_pattern[0]
            else:
                my_pattern = [pa for pa in my_pattern if my_onset in pa][0]
            this_file = my_pattern.format(
                subject=subject, run=run_label, kind=kind_map[data_type],
                context=context)
            files.append(op.join(path, this_file))

        elif output == 'evoked':
            # XXX add evoked template checks
            for condition in conditions:
                for sensor_mode in sensor_modes:
                    for diff_mode in diff_modes:
                        this_file = my_pattern.format(
                            subject=subject, kind=kind_map[data_type],
                            condition=condition, diff_mode=diff_mode,
                            sensor_mode=sensor_mode)
                        files.append(op.join(path, this_file))
        elif output == 'trial_info':
            this_file = my_pattern[0].format(
                subject=subject, run=run_label, kind=kind_map[data_type])
            files.append(op.join(path, this_file))
        elif data_type == 'meg_anatomy':
            path = file_map['path'].format(subject=subject)
            files.extend([op.join(path, pa.format(subject=subject))
                          for pa in my_pattern])
        elif data_type == 'freesurfer':
            path = file_map['path'].format(subject=subject)
            for pa in my_pattern:
                if mode == 'minimal':
                    if pa not in file_modes['freesurfer']['minimal']:
                        continue
                files.append(
                    op.join(path, output, pa.format(subject=subject)))
        else:
            raise ValueError('I never heard of `data_type` "%s".' % output)

    elif processing == 'unprocessed':
        if output == 'trial_info':
            raise ValueError('`trial_info` only exists for preprocessed data')
        path = unprocessed['path'].format(
            subject=subject, kind=kind_map[data_type], pipeline=pipeline,
            run=run_label)
        files.extend([op.join(path, p) for p in unprocessed['patterns']])

    else:
        raise ValueError('`processing` %s should be "unprocessed"'
                         ' or "preprocessed"')
    return [op.join(hcp_path, pa) for pa in files]
