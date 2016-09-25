.. _api_reference:

====================
Python API Reference
====================

This is the functions reference of MNE-HCP. Functions are
grouped thematically by analysis stage. Functions  that are not
below a module heading are found in the :py:mod:`hcp` namespace.

.. contents::
   :local:
   :depth: 2

.. currentmodule:: hcp

.. autosummary::
   :toctree: generated/
   :template: function.rst

   make_mne_anatomy
   compute_forward_stack


=================
Reading HCP files 
=================

:py:mod:`hcp.io`:

.. currentmodule:: hcp.io

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_raw_hcp
   read_epochs_hcp
   read_evokeds_hcp
   read_info_hcp
   read_annot_hcp
   read_ica_hcp
   read_trial_info_hcp
  
:py:mod:`hcp.io.file_mapping`:

.. currentmodule:: hcp.io.file_mapping

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_file_paths
   get_s3_keys_meg
   get_s3_keys_anatomy

=============================
Manipulating data and sensors
=============================

:py:mod:`hcp.preprocessing`:

.. currentmodule:: hcp.preprocessing

.. autosummary::
   :toctree: generated/
   :template: function.rst
 
   set_eog_ecg_channels
   apply_ica_hcp
   apply_ref_correction
   map_ch_coords_to_mne
   interpolate_missing


================
Visualizing data
================

:py:mod:`hcp.viz`:

.. currentmodule:: hcp.viz

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot_coregistration
   make_hcp_bti_layout