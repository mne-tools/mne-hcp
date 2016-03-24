# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import os.path as op
from copy import deepcopy

import mne

from scipy import sparse
from mne.externals.h5io import read_hdf5, write_hdf5

from ..io.readers import extract_anatomy
from ..io.readers import read_raw_hcp
from ..io.readers import read_info_hcp


def prepare_inverse_solution(file_map, hcp_path, anatomy_path,
                             n_jobs, subject,
                             recordings_path, noise_cov_filter=None,
                             fwd_params=None, src_params=None,
                             cov_mode='raw',
                             noise_cov_params=None,
                             fname=None):
    """"

    Convenience script for conducting standard MNE analyses.

    Parameters
    ----------
    file_map : dict
        The subejct records
    hcp_path : str
        The directory containing the HCP data.
    anatomy_path : str
        The directory containing the extracted HCP subject data.
    noise_cov_filter : None | dict
        The filter params to be used for the noise cov.
    fwd_params : None | dict
        The forward parameters
    src_params : None | dict
        The src params. Defaults to:

        dict(subject='fsaverage', fname=None, spacing='oct6', n_jobs=2,
             surface='white', subjects_dir=anatomy_path, add_dist=True)

    cov_mode : 'raw'
        The computational mode for covariance. Currently only 'raw' is
        supported.
    noise_cov_params : None | dict
        The parameters for the noise covariance computation.
    fname : None | str
        The file namae to be used for saving.
    """
    #  bake in support for precomuting h5 io
    extract_anatomy(
        file_map, hcp_path=hcp_path, anatomy_path=anatomy_path,
        recordings_path=recordings_path)
    head_mri_t = mne.read_trans(
        op.join(recordings_path, subject, '{}-head_mri-trans.fif'.format(
            subject)))
    meg_info = read_info_hcp(subject=file_map, hcp_path=hcp_path, run=1)[0]
    head_mri_t['from'] = mne.io.constants.FIFF.FIFFV_COORD_HEAD

    if cov_mode == 'raw' and noise_cov_params is None:
        noise_cov_params = _update_dict_defaults(
            noise_cov_params,
            dict(tmin=None, tmax=None, tstep=0.2, reject=None, flat=None,
                 picks=None, verbose=None))
    elif cov_mode != 'raw':
        raise NotImplementedError('Only cov_mode == "raw" is supported')

    src_params = _update_dict_defaults(
        src_params,
        dict(subject='fsaverage', fname=None, spacing='oct6', n_jobs=2,
             surface='white', subjects_dir=anatomy_path, add_dist=True))

    if fname and op.exists(fname):
        out = read_inverse_preprocessing(fname)
        src = out['src']
        src_morphed = out['src_morphed']
        bem_sol = out['bem_sol']
        fwd = out['fwd']
        noise_covs = out['noise_cov']
    else:
        raw_er = read_raw_hcp(
            subject=subject, hcp_path=hcp_path, kind='empty_room')

        raw_er.pick_types(meg=True, ref_meg=False)
        noise_cov = mne.compute_raw_covariance(raw_er)

        noise_covs = {'broad_band': noise_cov}

        if noise_cov_filter is not None:
            for band, params in noise_cov_filter:
                this_raw_er = raw_er.copy()
                for par in params:
                    this_raw_er.filter(n_jobs=n_jobs, **par)
                noise_covs[band] = mne.compute_raw_covariance(this_raw_er)
            del this_raw_er

        if src_params['add_dist']:
            src_params['add_dist'] = False
            add_source_space_distances = True
        src = mne.setup_source_space(**src_params)
        src_morphed = mne.morph_source_spaces(
            src, subject, subjects_dir=anatomy_path)

        if add_source_space_distances:
            src_morphed = mne.add_source_space_distances(
                src_morphed, n_jobs=n_jobs)

        bems = mne.make_bem_model(subject, conductivity=(0.3,),
                                  subjects_dir=anatomy_path,
                                  ico=None)
        bem_sol = mne.make_bem_solution(bems)
        meg_info = mne.pick_info(
            meg_info, [meg_info['ch_names'].index(k) for k in raw_er.ch_names])

        fwd = mne.make_forward_solution(
            meg_info, trans=head_mri_t, bem=bem_sol, src=src_morphed,
            n_jobs=n_jobs)
        if fname:
            write_inverse_preprocessing(
                fname=fname, fwd=fwd, src=src, src_morphed=src_morphed,
                bem_sol=bem_sol, noise_cov=noise_covs)

    return fwd, noise_covs, src, src_morphed, bem_sol, meg_info


def _sparse_dist_to_dict(inst, src_loc=None):
    """Helper to convert sparse matrix to dict"""
    inst = deepcopy(inst)
    if src_loc is not None:
        src = inst[src_loc]
    else:
        src = inst
    for ii, src_ in enumerate(src):
        dists = src_['dist']
        src[ii]['dist'] = {'data': (dists.data, dists.nonzero()),
                           'shape': dists.shape, 'dtype': 'f4'}
    return inst


def _update_dict_dist_to_sparse(src):
    """Helper to convert dict to sparse matrix"""
    for ii, src_, in enumerate(src):
        dists = src_['dist']
        src[ii]['dist'] = sparse.csr_matrix(
            dists['data'], shape=dists['shape'],
            dtype=dists['dtype'])


def write_inverse_preprocessing(
        fname, fwd, src, src_morphed, bem_sol, noise_cov):
    """Write workflow intermediate results"""
    write_hdf5(fname, src, title='hcp/mne/src', overwrite='update')

    write_hdf5(fname, _sparse_dist_to_dict(src_morphed),
               title='hcp/mne/src_morphed',
               overwrite='update')

    write_hdf5(fname, noise_cov, title='hcp/mne/noise_cov', overwrite='update')

    write_hdf5(fname, _sparse_dist_to_dict(fwd, src_loc='src'),
               title='hcp/mne/fwd', overwrite='update')

    write_hdf5(fname, bem_sol, title='hcp/mne/bem_sol', overwrite='update')

    write_hdf5(fname, ['hcp/mne/fwd', 'hcp/mne/src', 'hcp/mne/noise_cov',
                       'hcp/mne/src_morphed', 'hcp/mne/bem_sol'],
               title='hcp/fields', overwrite='update')


def read_inverse_preprocessing(fname):
    """Write workflow intermediate results"""
    fields = read_hdf5(fname, title='hcp/fields')
    out = dict()
    for field in fields:
        this_stuff = read_hdf5(fname, title=field)
        if field.endswith('cov'):
            for k, v in this_stuff.items():
                dim = v.pop('dim')
                kind = v.pop('kind')
                diag = bool(v.pop('diag'))
                this_stuff[k] = mne.Covariance(**v)
                this_stuff[k]['dim'] = dim
                this_stuff[k]['kind'] = kind
                this_stuff[k]['diag'] = diag

        elif 'src' in field:
            if isinstance(this_stuff[0]['dist'], dict):
                _update_dict_dist_to_sparse(this_stuff)
            this_stuff = mne.SourceSpaces(this_stuff)
        elif field.endswith('fwd'):
            if isinstance(this_stuff['src'][0]['dist'], dict):
                _update_dict_dist_to_sparse(this_stuff['src'])
            this_stuff['info'] = mne.io.meas_info.Info(this_stuff['info'])
            this_stuff = mne.forward.Forward(this_stuff)

        out[field.split('/')[-1]] = this_stuff
    return out


def _update_dict_defaults(values, defaults):
    """Helper to handle dict updates"""
    out = {k: v for k, v in defaults.items()}
    if isinstance(values, dict):
        out.update(values)
    return out
