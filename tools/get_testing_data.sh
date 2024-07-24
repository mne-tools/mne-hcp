#!/bin/bash

# On CIs, smaller files are downloaded and used (e.g., that have been cropped and
# decimated), see:
#
# https://github.com/dengemann/make-mne-hcp-testing-data/blob/master/make_mne_hcp_testing_data.ipynb
#
# Since the raw data are not decimated by a factor of 100, you need to set
# MNE_HCP_TEST_DECIM=1 when running the tests if you use these data directly
# (e.g., if you also set MNE_HCP_TESTING_PATH=~/mne-hcp-data/HCP).

set -exo pipefail

LOCAL=~/mne-hcp-data/HCP/105923
REMOTE=s3://hcp-openaccess/HCP/105923
mkdir -p $LOCAL/unprocessed/MEG
cd $LOCAL/unprocessed/MEG
s3cmd sync $REMOTE/unprocessed/MEG/3-Restin ./
s3cmd sync $REMOTE/unprocessed/MEG/4-Restin ./
s3cmd sync $REMOTE/unprocessed/MEG/5-Restin ./
s3cmd sync $REMOTE/unprocessed/MEG/8-StoryM ./
s3cmd sync $REMOTE/unprocessed/MEG/6-Wrkmem ./
s3cmd sync $REMOTE/unprocessed/MEG/7-Wrkmem ./
s3cmd sync $REMOTE/unprocessed/MEG/9-StoryM ./
s3cmd sync $REMOTE/unprocessed/MEG/10-Motort ./
s3cmd sync $REMOTE/unprocessed/MEG/11-Motort ./
s3cmd sync $REMOTE/unprocessed/MEG/1-Rnoise ./
cd $LOCAL/MEG
mkdir -p anatomy StoryM Wrkmem Motort Restin
cd anatomy
s3cmd sync $REMOTE/MEG/anatomy/105923_MEG_anatomy_transform.txt ./
s3cmd sync $REMOTE/MEG/anatomy/105923_MEG_anatomy_headmodel.mat ./
cd ../Restin
s3cmd sync $REMOTE/MEG/Restin/baddata ./
s3cmd sync $REMOTE/MEG/Restin/icaclass ./
s3cmd sync $REMOTE/MEG/Restin/rmegpreproc ./
cd ../StoryM
s3cmd sync $REMOTE/MEG/StoryM/baddata ./
s3cmd sync $REMOTE/MEG/StoryM/icaclass ./
s3cmd sync $REMOTE/MEG/StoryM/tmegpreproc ./
s3cmd sync $REMOTE/MEG/StoryM/eravg ./
cd ../Wrkmem
s3cmd sync $REMOTE/MEG/Wrkmem/baddata ./
s3cmd sync $REMOTE/MEG/Wrkmem/icaclass ./
s3cmd sync $REMOTE/MEG/Wrkmem/tmegpreproc ./
s3cmd sync $REMOTE/MEG/Wrkmem/eravg ./
cd ../Motort
s3cmd sync $REMOTE/MEG/Motort/baddata ./
s3cmd sync $REMOTE/MEG/Motort/icaclass ./
s3cmd sync $REMOTE/MEG/Motort/tmegpreproc ./
s3cmd sync $REMOTE/MEG/Motort/eravg ./
cd $LOCAL
mkdir -p T1w/
s3cmd sync $REMOTE/T1w/105923 ./T1w/
