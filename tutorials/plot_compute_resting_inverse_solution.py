"""
.. _tut_compute_inverse_erf:

========================================
Compute inverse solution for evoked data
========================================

Here we'll use our knowledge from the other examples and tutorials
to compute an inverse solution and apply it on resting state data.
"""
# Author: Denis A. Engemann
#         Luke Bloy <luke.bloy@gmail.com>
#         Eric Larson <larson.eric.d@gmail.com>
# License: BSD 3 clause

import os.path as op
import mne
from mne.filter import next_fast_len
import hcp
from hcp import preprocessing as preproc

##############################################################################
# we assume our data is inside a designated folder under $HOME
storage_dir = op.expanduser('~/mne-hcp-data')
hcp_path = op.join(storage_dir, 'HCP')
recordings_path = op.join(storage_dir, 'hcp-meg')
subjects_dir = op.join(storage_dir, 'hcp-subjects')
subject = '105923'  # our test subject
data_type = 'rest'
run_index = 0

##############################################################################
# We're reading the evoked data.
# These are the same as in :ref:`tut_plot_evoked`

raw = hcp.read_raw(subject=subject,
                   data_type=data_type, hcp_path=hcp_path,
                   run_index=run_index)
raw.load_data()
raw.crop(0, 250)
raw.resample(100.)
raw.info['bads'] = ['A147']  # we know this one is bad here.
preproc.set_eog_ecg_channels(raw)
preproc.apply_ref_correction(raw)

##############################################################################
# Remap BTi sensor coordinates to Neuromag for plotting.
info = raw.info.copy()
preproc.map_ch_coords_to_mne(raw)  # we have to revert it later for inverse!


raw.info['projs'] = []
ecg_ave = mne.preprocessing.create_ecg_epochs(raw).average()
ecg_ave.plot_joint(times='peaks')
# We see no clear ECG artifact!

eog_ave = mne.preprocessing.create_eog_epochs(raw).average()
eog_ave.plot_joint(times=[0])
# But we see a clear EOG artifact. So we'll compute an SSP

ssp_eog, _ = mne.preprocessing.compute_proj_eog(
    raw, n_grad=1, n_mag=1, average=True, reject=dict(mag=5e-12))
raw.add_proj(ssp_eog, remove_existing=True)

fig = mne.viz.plot_projs_topomap(raw.info['projs'],
                                 info=raw.info)
fig.suptitle('HCP BTi/4D SSP')
fig.subplots_adjust(0.05, 0.05, 0.95, 0.85)

##############################################################################
# Explore data

n_fft = next_fast_len(int(round(4 * raw.info['sfreq'])))
print('Using n_fft=%d (%0.1f sec)' % (n_fft, n_fft / raw.info['sfreq']))
fig = raw.plot_psd(n_fft=n_fft, proj=True, fmax=50)
fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)

# We see clear alpha and beta peaks!

##############################################################################
# We'll now use a convenience function to get our forward and source models
# instead of computing them by hand.

hcp.make_mne_anatomy(
    subject, subjects_dir=storage_dir + '/hcp-subjects',
    hcp_path=storage_dir + '/HCP',recordings_path=storage_dir + '/hcp-meg')

src_outputs = hcp.anatomy.compute_forward_stack(
    subject=subject, subjects_dir=subjects_dir,
    hcp_path=hcp_path, recordings_path=recordings_path,
    # speed up computations here. Setting `add_dist` to True may improve the
    # accuracy.
    src_params=dict(add_dist=False),
    info_from=dict(data_type=data_type, run_index=run_index))

fwd = src_outputs['fwd']

##############################################################################
# Now we can compute the noise covariance. For this purpose we will apply
# the same filtering as was used for the computations of the ERF in the first
# place. See also :ref:`tut_reproduce_erf`.

raw_noise = hcp.read_raw(subject=subject, hcp_path=hcp_path,
                         data_type='noise_empty_room')
raw_noise.load_data()

# apply ref channel correction, drop ref channels, add ssp.
preproc.apply_ref_correction(raw_noise)
raw_noise.add_proj(ssp_eog)


##############################################################################
# Note that using the empty room noise covariance will inflate the SNR of the
# evkoked and renders comparisons  to `baseline` rather uninformative.
noise_cov = mne.compute_raw_covariance(raw_noise, method='oas')


##############################################################################
# Compute and apply inverse to PSD estimated using multitaper + Welch.
# Group into frequency bands, then normalize each source point and sensor
# independently. This makes the value of each sensor point and source location
# in each frequency band the percentage of the PSD accounted for by that band.

freq_bands = dict(
    delta=(2, 4), theta=(5, 7), alpha=(8, 12), beta=(15, 29), gamma=(30, 45))
topos = dict()
stcs = dict()

##############################################################################
# redo original BTi sensor coordinates.
raw.info = info

inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, forward=fwd, noise_cov=noise_cov, verbose=True)

stc_psd, ch_psd = mne.minimum_norm.compute_source_psd(
    raw, inverse_operator, lambda2=1. / 9.,
    n_fft=n_fft, dB=False, return_sensor=True, verbose=True,
    fmax=50)

# remap sensors coords for power
preproc.map_ch_coords_to_mne(ch_psd)

stc_psd = stc_psd.to_original_src(
    src_outputs['src_fsaverage'], subjects_dir=subjects_dir)

topo_norm = ch_psd.data.sum(axis=1, keepdims=True)
stc_norm = stc_psd.sum()
# Normalize each source point by the total power across freqs
for band, limits in freq_bands.items():
    data = ch_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
    topos[band] = mne.EvokedArray(
        100 * data / topo_norm, ch_psd.info)
    stcs[band] = \
        100 * stc_psd.copy().crop(*limits).sum() / stc_norm.data


###############################################################################
# Now we can make some plots of each frequency band. Note that the OPM head
# coverage is only over right motor cortex, so only localization
# of beta is likely to be worthwhile.
#
# Theta
# -----


def plot_band(band):
    """Plot channel and source band power."""
    title = "%s\n(%d-%d Hz)" % ((band,) + freq_bands[band])
    topos[band].plot_topomap(
        times=0., scalings=1., cbar_fmt='%0.1f', vmin=0, cmap='inferno',
        time_format=title)
    brain = stcs[band].plot(
        subject='fsaverage', subjects_dir=subjects_dir, views='cau', hemi='both',
        time_label=title, title=title, colormap='inferno',
        clim=dict(kind='percent', lims=(70, 85, 99)))
    brain.show_view(dict(azimuth=0, elevation=0), roll=0)
    return fig, brain


fig_theta, brain_theta = plot_band('theta')

###############################################################################
# Alpha
# -----

fig_alpha, brain_alpha = plot_band('alpha')

###############################################################################
# Beta
# ----

fig_beta, brain_beta = plot_band('beta')

###############################################################################
# Gamma
# -----

fig_gamma, brain_gamma = plot_band('gamma')
