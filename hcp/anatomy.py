import os
import os.path as op
import re

import numpy as np
from scipy import linalg

import mne
from mne.io.pick import _pick_data_channels, pick_info
from mne import write_trans, write_surface
from mne.transforms import Transform, apply_trans
from mne.utils import logger

from .io.file_mapping import get_file_paths
from .io.read import _read_trans_hcp
from .io.read import _get_head_model
from .io import read_info_hcp


def make_mne_anatomy(subject, anatomy_path, recordings_path=None,
                     hcp_path=op.curdir, mode='minimal', outputs=(
                         'label', 'mri', 'surf')):
    """Extract relevant anatomy and create MNE friendly directory layout

    The function will create the following outputs by default:

    $anatomy_path/$subject/bem/inner_skull.surf
    $anatomy_path/$subject/label/*
    $anatomy_path/$subject/mri/*
    $anatomy_path/$subject/surf/*
    $recordings_path/$subject/$subject-head_mri-trans.fif

    These can then be set as $SUBJECTS_DIR and as MEG directory, consistent
    with MNE examples.

    Parameters
    ----------
    subject : str
        The subject name.
    anatomy_path : str
        The path corresponding to MNE/freesurfer SUBJECTS_DIR (to be created)
    hcp_path : str
        The path where the HCP files can be found.
    mode : {'minimal', 'full'}
        If 'minimal', only the directory structure is created. If 'full' the
        freesurfer outputs shipped withm HCP (see `outputs`) are symbolically
        linked.
    outputs : {'label', 'mri', 'stats', 'surf', 'touch'}
        The outputs of the freesrufer pipeline shipped by HCP. Defaults to
        ('mri', 'surf'), the minimum needed to extract MNE-friendly anatomy
        files and data.
    """
    if mode not in ('full', 'minimal'):
        raise ValueError('`mode` must either be "minimal" or "full"')
    if hcp_path == op.curdir:
        hcp_path = op.realpath(hcp_path)
    if not op.isabs(anatomy_path):
        anatomy_path = op.realpath(anatomy_path)

    this_anatomy_path = op.join(anatomy_path, subject)
    if not op.isabs(recordings_path):
        recordings_path = op.realpath(recordings_path)

    this_recordings_path = op.join(recordings_path, subject)

    if not op.exists(this_recordings_path):
        os.makedirs(this_recordings_path)

    for output in outputs:
        if not op.exists(op.join(this_anatomy_path, output)):
            os.makedirs(op.join(this_anatomy_path, output))
        if output == 'mri':
            for suboutput in ['orig', 'transforms']:
                if not op.exists(
                        op.join(this_anatomy_path, output, suboutput)):
                    os.makedirs(op.join(this_anatomy_path, output, suboutput))

        files = get_file_paths(
            subject=subject, data_type='freesurfer', output=output,
            mode=mode, hcp_path=hcp_path)
        for source in files:
            match = [match for match in re.finditer(subject, source)][-1]
            split_path = source[:match.span()[1] + 1]
            target = op.join(this_anatomy_path, source.split(split_path)[-1])
            if not op.isfile(target) and not op.islink(target):
                os.symlink(source, target)

    logger.info('reading extended structural processing ...')

    # Step 1 #################################################################
    # transform head models to expected coordinate system

    # make hcp trans
    transforms_fname = get_file_paths(
        subject=subject, data_type='meg_anatomy', output='transforms',
        hcp_path=hcp_path)
    transforms_fname = [k for k in transforms_fname if
                        k.endswith('transform.txt')][0]
    hcp_trans = _read_trans_hcp(fname=transforms_fname, convert_to_meter=False)

    # get RAS freesurfer trans
    c_ras_trans_fname = get_file_paths(
        subject=subject, data_type='freesurfer', output='mri',
        hcp_path=hcp_path)
    c_ras_trans_fname = [k for k in c_ras_trans_fname if
                         k.endswith('c_ras.mat')][0]
    logger.info('reading RAS freesurfer transform')
    # ceci n'est pas un .mat file ...

    with open(op.join(anatomy_path, c_ras_trans_fname)) as fid:
        ras_trans = np.array([
            r.split() for r in fid.read().split('\n') if r],
            dtype=np.float64)

    logger.info('Combining RAS transform and coregistration')
    ras_trans_m = linalg.inv(ras_trans)  # and the inversion

    logger.info('extracting head model')
    head_model_fname = get_file_paths(
        subject=subject, data_type='meg_anatomy', output='head_model',
        hcp_path=hcp_path)[0]
    pnts, faces = _get_head_model(head_model_fname=head_model_fname)

    logger.info('coregistring head model to MNE-HCP coordinates')
    pnts = apply_trans(ras_trans_m.dot(hcp_trans['bti2spm']), pnts)

    tri_fname = op.join(this_anatomy_path, 'bem', 'inner_skull.surf')
    if not op.exists(op.dirname(tri_fname)):
        os.makedirs(op.dirname(tri_fname))
    write_surface(tri_fname, pnts, faces)

    # Step 2 #################################################################
    # write corresponding device to MRI transform

    logger.info('extracting coregistration')
    # now convert to everything meter too here
    ras_trans_m[:3, 3] *= 1e-3
    bti2spm = hcp_trans['bti2spm']
    bti2spm[:3, 3] *= 1e-3
    head_mri_t = Transform(  # we're lying here for a good purpose
        'head', 'mri', np.dot(ras_trans_m, bti2spm))  # it should be 'ctf_head'
    write_trans(
        op.join(this_recordings_path, '%s-head_mri-trans.fif') % subject,
        head_mri_t)


def make_forward_stack(anatomy_path,
                       subject,
                       recordings_path,
                       info_from=(('data_type', 'rest'), ('run_index', 0)),
                       fwd_params=None, src_params=None,
                       hcp_path=op.curdir, n_jobs=1):
    """
    Convenience function for conducting standard MNE analyses.

    .. note::
       this function computes bem solutions, source spaces and forward models
       optimized for connectivity computation, i.e., the fsaverage space
       is morphed onto the subject's space.

    Parameters
    ----------
    subject : str
        The subject name.
    hcp_path : str
        The directory containing the HCP data.
    recordings_path : str
        The path where MEG data and transformations are stored.
    anatomy_path : str
        The directory containing the extracted HCP subject data.
    info_from : tuple of tuples | dict
        The reader info concerning the data from which sensor positions
        should be read.
        Must not be empty room as sensor positions are in head
        coordinates for 4D systems, hence not available in that case.
        Note that differences between the sensor positions across runs
        are smaller than 12 digits, hence negligible.
    fwd_params : None | dict
        The forward parameters
    src_params : None | dict
        The src params. Defaults to:

        dict(subject='fsaverage', fname=None, spacing='oct6', n_jobs=2,
             surface='white', subjects_dir=anatomy_path, add_dist=True)
    hcp_path : str
        The prefix of the path of the HCP data.
    n_jobs : int
        The number of jobs to use in parallel.

    Returns
    -------
    out : dict
        A dictionary with the following keys:
            fwd : instance of mne.Forward
                The forward solution.
            src_subject : instance of mne.SourceSpace
                The source model on the subject's surface
            src_fsaverage : instance of mne.SourceSpace
                The source model on fsaverage's surface
            bem_sol : dict
                The BEM.
            info : instance of mne.io.meas_info.Info
                The actual measurement info used.
    """
    if isinstance(info_from, tuple):
        info_from = dict(info_from)

    head_mri_t = mne.read_trans(
        op.join(recordings_path, subject, '{}-head_mri-trans.fif'.format(
            subject)))

    src_params = _update_dict_defaults(
        src_params,
        dict(subject='fsaverage', fname=None, spacing='oct6', n_jobs=n_jobs,
             surface='white', subjects_dir=anatomy_path, add_dist=True))

    add_source_space_distances = False
    if src_params['add_dist']:  # we want the distances on the morphed space
        src_params['add_dist'] = False
        add_source_space_distances = True

    src_fsaverage = mne.setup_source_space(**src_params)
    src_subject = mne.morph_source_spaces(
        src_fsaverage, subject, subjects_dir=anatomy_path)

    if add_source_space_distances:  # and here we compute them post hoc.
        src_subject = mne.add_source_space_distances(
            src_subject, n_jobs=n_jobs)

    bems = mne.make_bem_model(subject, conductivity=(0.3,),
                              subjects_dir=anatomy_path,
                              ico=None)  # ico = None for morphed SP.
    bem_sol = mne.make_bem_solution(bems)
    bem_sol['surfs'][0]['coord_frame'] = 5

    info = read_info_hcp(subject=subject, hcp_path=hcp_path, **info_from)
    picks = _pick_data_channels(info, with_ref_meg=False)
    info = pick_info(info, picks)

    # here we assume that as a result of our MNE-HCP processing
    # all other transforms in info are identity
    for trans in ['dev_head_t', 'ctf_head_t']:
        #  'dev_ctf_t' is not identity
        assert np.sum(info[trans]['trans'] - np.eye(4)) == 0

    surf = src_params['surface']
    add_dist = add_source_space_distances
    spacing = src_params['spacing']
    src_type = 'subject_on_fsaverage'

    fwd = mne.make_forward_solution(
        info, trans=head_mri_t, bem=bem_sol, src=src_subject,
        n_jobs=n_jobs)

    mne.write_forward_solution(
        op.join(recordings_path, subject, '%s-%s-%s-%s-fwd.fif' % (
                surf, spacing, add_dist, src_type)),
        fwd, overwrite=True)

    mne.write_source_spaces(
        op.join(recordings_path, subject, '%s-%s-%s-%s-src.fif' % (
                surf, spacing, add_dist, src_type)),
        src_subject)

    fname_bem = op.join(
        anatomy_path, subject, 'bem', '%s-%i-bem.fif' % (
            subject, bem_sol['solution'].shape[0]))
    mne.write_bem_solution(fname_bem, bem_sol)

    return dict(fwd=fwd, src_subject=src_subject,
                src_fsaverage=src_fsaverage,
                bem_sol=bem_sol, info=info)


def _update_dict_defaults(values, defaults):
    """Helper to handle dict updates"""
    out = {k: v for k, v in defaults.items()}
    if isinstance(values, dict):
        out.update(values)
    return out
