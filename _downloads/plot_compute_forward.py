"""
.. _tut_forward:

=====================
Compute forward model
=====================

Here we'll first compute a source space, then the bem model
and finally the forward solution.
"""
# Author: Denis A. Enegemann
# License: BSD 3 clause

import os.path as op
import mne
import hcp

##############################################################################
# we assume our data is inside a designated folder under $HOME
storage_dir = op.expanduser('~/mne-hcp-data')
hcp_path = op.join(storage_dir, 'HCP')
recordings_path = op.join(storage_dir, 'hcp-meg')
subjects_dir = op.join(storage_dir, 'hcp-subjects')
subject = '105923'  # our test subject

##############################################################################
# and we assume to have the downloaded data, the MNE/freesurfer style
# anatomy directory, and the MNE style MEG directory.
# these can be obtained from :func:`make_mne_anatomy`.
# See also :ref:`tut_make_anatomy`.

##############################################################################
# first we read the coregistration.

head_mri_t = mne.read_trans(
    op.join(recordings_path, subject, '{}-head_mri-trans.fif'.format(
            subject)))

##############################################################################
# Now we can setup our source model.
# Note that spacing has to be set to 'all' since no common MNE resampling
# scheme has been employed in the HCP pipelines.
# Since this will take very long time to compute and at this point no other
# decimation scheme is available inside MNE, we will compute the source
# space on fsaverage, the freesurfer average brain, and morph it onto
# the subject's native space. With `oct6` we have ~8000 dipole locations.

src_fsaverage = mne.setup_source_space(
    subject='fsaverage', subjects_dir=subjects_dir, add_dist=False,
    spacing='oct6', overwrite=True)

# now we morph it onto the subject.

src_subject = mne.morph_source_spaces(
    src_fsaverage, subject, subjects_dir=subjects_dir)

##############################################################################
# For the same reason `ico` has to be set to `None` when computing the bem.
# The headshape is not computed with MNE and has a none standard configuration.

bems = mne.make_bem_model(subject, conductivity=(0.3,),
                          subjects_dir=subjects_dir,
                          ico=None)  # ico = None for morphed SP.
bem_sol = mne.make_bem_solution(bems)
bem_sol['surfs'][0]['coord_frame'] = 5

##############################################################################
# Now we can read the channels that we want to map to the cortical locations.
# Then we can compute the forward solution.

info = hcp.read_info(subject=subject, hcp_path=hcp_path, data_type='rest',
                     run_index=0)

picks = mne.pick_types(info, meg=True, ref_meg=False)
info = mne.pick_info(info, picks)

fwd = mne.make_forward_solution(info, trans=head_mri_t, bem=bem_sol,
                                src=src_subject)
mag_map = mne.sensitivity_map(
    fwd, projs=None, ch_type='mag', mode='fixed', exclude=[], verbose=None)

##############################################################################
# we display sensitivity map on the original surface with little smoothing
# and admire the expected curvature-driven sensitivity pattern.

mag_map = mag_map.to_original_src(src_fsaverage, subjects_dir=subjects_dir)
mag_map.plot(subject='fsaverage', subjects_dir=subjects_dir,
             clim=dict(kind='percent', lims=[0, 50, 99]),
             smoothing_steps=2)
