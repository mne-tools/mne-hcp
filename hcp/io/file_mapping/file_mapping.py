import os.path as op

"""Notes

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
            ('epochs', 'rmeg'): [
                '{subject}_MEG_{run}-{kind}_{context}preproc.mat'],
            ('epochs', 'tmeg'): [
                '{subject}_MEG_{run}-{kind}_tmegpreproc_{onset}.mat',
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
                (r'{subject}_MEG_{kind}_eravg_[{condition}]_{diff_modes}_'
                 '[{sensor_mode}].mat')
                ],
            'tfr': [
                (r'{subject}_MEG_{kind}_tfavg_[{condition}]_{diff_modes}_'
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


evoked_map = {
    'modes': {'planar': 'MODE-planar', 'mag': 'MODE-mag'},
    # always BT-diff
    'task_motor': (
        'LM-TEMG-LF',
        'LM-TEMG-LH',
        'LM-TEMG-RF',
        'LM-TEMG-RH',
        'LM-TFLA-LF',
        'LM-TFLA-LH',
        'LM-TFLA-RF',
        'LM-TFLA-RH'),
    # if versus in name then OP-diff + BT-diff, else BT-diff
    'task_working_memory': (
        'LM-TIM-0B',
        'LM-TIM-0B-versus-2B',
        'LM-TIM-2B',
        'LM-TIM-face',
        'LM-TIM-face-versus-tool',
        'LM-TIM-tool',
        'LM-TRESP-0B',
        'LM-TRESP-0B-versus-2B',
        'LM-TRESP-2B',
        'LM-TRESP-face',
        'LM-TRESP-face-versus-tool',
        'LM-TRESP-tool'),
    # if versus in name then OP-diff + BT-diff, else BT-diff
    'task_story_math': (
        'LM-TEV-mathnumopt',
        'LM-TEV-mathnumoptcor-versus-mathnumoptwro',
        'LM-TEV-mathnumque',
        'LM-TEV-mathnumque-versus-mathoper',
        'LM-TEV-mathnumquelate-versus-mathnumqueearly',
        'LM-TEV-mathoper',
        'LM-TEV-mathsentnon',
        'LM-TEV-storoptcor-versus-storoptwro',
        'LM-TEV-storsentnon',
        'LM-TEV-storsentnon-versus-mathsentnon',
        'LM-TRESP-all')
}


freesurfer_files = op.join(op.dirname(__file__), 'data', '%s.txt')
for kind, patterns in preprocessed['freesurfer']['patterns'].items():
    with open(freesurfer_files % kind) as fid:
        patterns.extend([k.rstrip('\n') for k in fid.readlines()])

pipeline_map = {
    'ica': 'icaclass',
    'bads': 'baddata',
    'psd': 'powavg',
    'evoked': 'eravg',
    'tfr': 'tfavg',
    'trial_info': 'tmegpreproc'
}

kind_map = {
    'task_motor': 'Motort',
    'task_working_memory': 'Wrkmem',
    'task_story_math': 'StoryM',
    'rest': 'Restin',
    'noise_empty_room': 'Rnoise',
    'noise_subject': 'Pnoise',
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


def _map_onset(onset, data_type, output):
    """Helper to resolve stim and resp according to context"""
    out = onset
    if data_type == 'task_working_memory':
        out = {'stim': 'TIM', 'resp': 'TRESP'}[onset]
    elif data_type == 'task_motor':
        out = {'stim': 'TFLA', 'resp': 'TEMG'}[onset]
    elif data_type == 'task_story_math' and output == 'evoked':
        out = {'stim': 'TEV', 'resp': 'TRESP'}[onset]
    elif data_type == 'task_story_math' and output == 'epochs':
        out = {'stim': 'TEV', 'resp': 'TRESP', 'sentence': 'BSENT',
               'block': 'BUN'}[onset]
    return out


def _map_diff_mode(condition, data_type):
    """Helper to resolve diff mode according to context"""
    diff_mode = '[BT-diff]'
    if 'versus' in condition:
        diff_mode = '[OP-diff]_[BT-diff]'
    return diff_mode


def get_file_paths(subject, data_type, output, run_index=0,
                   onset='stim',
                   sensor_mode='mag', hcp_path='.'):
    """This is the MNE-HCP file path synthesizer

    An easy conceptual mapper from questions to file paths

    Parameters
    ----------
    subject : str
        The subject, a 6 digit code.
    data_type : str
        The acquisition context of the data. The following ones are supported:
        'rest'
        'noise'
        'task_motor'
        'task_story_math'
        'task_working_memory'
    output : str
        The kind of output. The following ones are supported:
        'raw',
        'epochs'
        'evoked'
        'ica'
        'annotations'
        'trial_info'
        'freesurfer'
        'meg_anatomy'
    onset : {'stim', 'resp', 'sentence', 'block'}
        The event onset. Only considered for epochs and evoked outputs
        The mapping is generous, everything that is not a response is a
        stimulus, in the sense of internal or external events. sentence and
        block are specific to task_story_math.
    sensor_mode : {'mag', 'planar'}
        The sensor projection. Defaults to 'mag'. Only relevant for
        evoked output.
    run_index : int
        The run index. For the first run, use 0, for the second, use 1.
        Also see HCP documentation for the number of runs for a given data
        type.
    hcp_path : str
        The HCP directory, defaults to op.curdir.

    Returns
    -------
    out : list of str
        The file names.
    """
    if data_type not in kind_map:
        raise ValueError('I never heard of `%s` -- are you sure this is a'
                         ' valid HCP type? I currenlty support:\n%s' % (
                             data_type, ' \n'.join(
                                 [k for k in kind_map if '_' in k])))
    context = ('rmeg' if 'rest' in data_type else 'tmeg')
    sensor_mode = evoked_map['modes'][sensor_mode]
    my_onset = _map_onset(onset, data_type, output)
    if data_type not in ('meg_anatomy', 'freesurfer'):
        my_runs = run_map[data_type]
        if run_index >= len(my_runs):
            raise ValueError('For `data_type=%s` we have %d runs. '
                             'You asked for run index %d.' % (
                                 data_type, len(my_runs), run_index))
        run_label = my_runs[run_index]
    else:
        run_label = None
    if (data_type in ('noise_subject',
                      'noise_empty_room') and output in
            ('epochs', 'evoked', 'ica', 'annot')):
        raise ValueError('You requested preprocessed data of type "%s" '
                         'and output "%s". HCP does not provide these data' %
                         (data_type, output))
    if (data_type in ('rest', 'noise_subject', 'noise_empty_room') and
            output in ('trial_info', 'evoked')):
        raise ValueError('%s not defined for %s' % (output, data_type))

    files = list()
    pipeline = pipeline_map.get(output, output)
    processing = 'preprocessed'
    if output == 'raw':
        processing = 'unprocessed'

    if processing == 'preprocessed':
        file_map = preprocessed[(data_type if data_type in (
                                 'meg_anatomy', 'freesurfer') else 'meg')]
        path = file_map['path'].format(
            subject=subject,
            pipeline=(context + 'preproc' if output == 'epochs'
                      else pipeline),
            kind=kind_map[data_type])

        if output == 'epochs':
            pattern_key = (output, context)
        else:
            pattern_key = output

        my_pattern = file_map['patterns'].get(pattern_key, None)
        if my_pattern is None:
            raise ValueError('What is output "%s"? I don\'t know about this.' %
                             output)

        if output in ('bads', 'ica'):
            files.extend(
                [op.join(path,
                         p.format(subject=subject, run=run_label,
                                  kind=kind_map[data_type]))
                 for p in my_pattern])

        elif output == 'epochs':
            my_pattern = my_pattern[0]
            formats = dict(
                subject=subject, run=run_label, kind=kind_map[data_type],
                context=context)
            if context != 'rest':
                formats.update(onset=my_onset)
            this_file = my_pattern.format(**formats)
            files.append(op.join(path, this_file))

        elif output == 'evoked':
            # XXX add evoked template checks
            for condition in evoked_map[data_type]:
                if my_onset not in condition:
                    continue
                this_file = my_pattern[0].format(
                    subject=subject, kind=kind_map[data_type],
                    condition=condition,
                    diff_modes=_map_diff_mode(condition, data_type),
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
                files.append(
                    op.join(path, output, pa.format(subject=subject)))
        else:
            raise ValueError('I never heard of `output` "%s".' % output)

    elif processing == 'unprocessed':
        path = unprocessed['path'].format(
            subject=subject, kind=kind_map[data_type], pipeline=pipeline,
            run=run_label)
        files.extend([op.join(path, p) for p in unprocessed['patterns']])

    else:
        raise ValueError('`processing` %s should be "unprocessed"'
                         ' or "preprocessed"')
    return [op.join(hcp_path, pa) for pa in files]
