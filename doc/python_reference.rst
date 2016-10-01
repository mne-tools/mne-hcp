.. _api_reference:

====================
Python API Reference
====================

This is the functions reference of MNE-HCP. Functions are
grouped thematically by analysis purpose. Functions  that are not
below a module heading are found in the :py:mod:`hcp` namespace.

.. contents::
   :local:
   :depth: 2

.. currentmodule:: hcp

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_raw
   read_epochs
   read_evokeds
   read_info
   read_annot
   read_ica
   read_trial_info

   make_mne_anatomy
   compute_forward_stack

==================================
Handling HCP files for downloading
==================================

.. currentmodule:: file_mapping

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
