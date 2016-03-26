# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import os.path as op

import numpy as np

import mne
from mne.io.pick import _pick_data_channels, pick_info

from ..io import read_info_hcp


def make_mne_forward(anatomy_path,
                     subject,
                     recordings_path,
                     fwd_params=None, src_params=None,
                     n_jobs=1, hcp_path=op.curdir):
    """"
    Convenience script for conducting standard MNE analyses.

    Parameters
    ----------
    subject : str
        The subject name.
    hcp_path : str
        The directory containing the HCP data.
    anatomy_path : str
        The directory containing the extracted HCP subject data.
    fwd_params : None | dict
        The forward parameters
    src_params : None | dict
        The src params. Defaults to:

        dict(subject='fsaverage', fname=None, spacing='oct6', n_jobs=2,
             surface='white', subjects_dir=anatomy_path, add_dist=True)
    """

    head_mri_t = mne.read_trans(
        op.join(recordings_path, subject, '{}-head_mri-trans.fif'.format(
            subject)))

    src_params = _update_dict_defaults(
        src_params,
        dict(subject='fsaverage', fname=None, spacing='oct6', n_jobs=2,
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

    info = read_info_hcp(
        subject=subject, data_type='noise_empty_room', hcp_path=hcp_path)
    picks = _pick_data_channels(info, with_ref_meg=False)
    info = pick_info(info, picks)

    # here we assume that as a result of our MNE-HCP processing
    # all other transforms in info are identity
    for trans in ['dev_head_t', 'dev_ctf_t', 'ctf_head_t']:
        assert np.sum(info[trans]['trans'] - np.eye(4)) == 0

    fwd = mne.make_forward_solution(
        info, trans=head_mri_t, bem=bem_sol, src=src_subject,
        n_jobs=n_jobs)

    return dict(fwd=fwd, src=src_subject, src_fsaverage=src_fsaverage,
                bem_sol=bem_sol, info=info)


def _update_dict_defaults(values, defaults):
    """Helper to handle dict updates"""
    out = {k: v for k, v in defaults.items()}
    if isinstance(values, dict):
        out.update(values)
    return out
