import os
import os.path as op
import shutil

import pytest

import mne
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from numpy.testing import assert_array_equal

import hcp
from hcp.io.read import _hcp_pick_info


run_inds=[0, 1, 2]
max_runs = 3  # from the test dataset creation step
bti_chans = {"A" + str(i) for i in range(1, 249, 1)}
task_types = pytest.mark.parametrize("data_type", ["task_story_math", "task_working_memory", "task_motor"])
run_indices = pytest.mark.parametrize("run_index", run_inds[: max_runs])
run_indices_2 = pytest.mark.parametrize("run_index", run_inds[: max_runs][:2])

test_decim = int(os.getenv("MNE_HCP_TEST_DECIM", "100"))
sfreq_preproc = 508.63 / test_decim
sfreq_raw = 2034.5101
#lowpass_preproc = 150
#highpass_preproc = 1.3

epochs_bounds = {
    "task_motor": (-1.2, 1.2),
    "task_working_memory": (-1.5, 2.5),
    "task_story_math": (-1.5, 4),
    "rest": (0, 2),
}


@run_indices
def test_read_annot(hcp_params, run_index):
    """Test reading annotations."""
    annots = hcp.read_annot(data_type="rest", run_index=run_index, **hcp_params)
    # channels
    assert_equal(
        list(sorted(annots["channels"])),
        ["all", "ica", "manual", "neigh_corr", "neigh_stdratio"],
    )
    for channels in annots["channels"].values():
        for chan in channels:
            assert chan in bti_chans

    # segments
    assert_equal(
        list(sorted(annots["ica"])),
        [
            "bad",
            "brain_ic",
            "brain_ic_number",
            "brain_ic_vs",
            "brain_ic_vs_number",
            "ecg_eog_ic",
            "flag",
            "good",
            "physio",
            "total_ic_number",
        ],
    )
    for components in annots["ica"].values():
        if len(components) > 0:
            assert min(components) >= 0
            assert max(components) <= 248


def _basic_raw_checks(raw):
    """Helper for testing raw files"""
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    assert_equal(len(picks), 248)
    ch_names = [raw.ch_names[pp] for pp in picks]
    assert all(ch.startswith("A") for ch in ch_names)
    ch_sorted = list(sorted(ch_names))
    assert ch_sorted != ch_names
    assert_equal(np.round(raw.info["sfreq"], 4), sfreq_raw)


@run_indices
def test_read_raw_rest(hcp_params, run_index):
    """Test reading raw for resting state"""
    raw = hcp.read_raw(data_type="rest", run_index=run_index, **hcp_params)
    _basic_raw_checks(raw=raw)


@task_types
@run_indices
def test_read_raw_task(hcp_params, data_type, run_index):
    """Test reading raw for tasks"""
    if run_index == 2:
        assert_raises(
            ValueError,
            hcp.read_raw,
            data_type=data_type,
            run_index=run_index,
            **hcp_params,
        )
        return
    raw = hcp.read_raw(data_type=data_type, run_index=run_index, **hcp_params)
    _basic_raw_checks(raw=raw)


@pytest.mark.parametrize("data_type", ["noise_empty_room"])
@run_indices_2
def test_read_raw_noise(hcp_params, data_type, run_index):
    """Test reading raw for empty room noise"""
    if run_index == 1:
        assert_raises(
            ValueError,
            hcp.read_raw,
            data_type=data_type,
            run_index=run_index,
            **hcp_params,
        )
        return
    raw = hcp.read_raw(data_type=data_type, run_index=run_index, **hcp_params)
    _basic_raw_checks(raw=raw)


def _epochs_basic_checks(epochs, annots, *, data_type):
    n_good = 248 - len(annots["channels"]["all"])
    if data_type == "task_motor":
        n_good += 4
    assert_equal(len(epochs.ch_names), n_good)
    assert_allclose(epochs.info["sfreq"], sfreq_preproc, rtol=1e-4)
    assert_array_equal(np.unique(epochs.events[:, 2]), np.array([99], dtype=np.int64))
    assert (
        _check_bounds(
            epochs.times,
            epochs_bounds[data_type],
            atol=1.0 / epochs.info["sfreq"],
        )  # decim tolerance
    )

    # XXX these seem not to be reliably set. checkout later.
    # assert_equal(
    #     epochs.info['lowpass'],
    #     lowpass_preproc)
    # assert_equal(
    #     epochs.info['highpass'],
    #     highpass_preproc)


@run_indices_2
def test_read_epochs_rest(hcp_params, run_index):
    """Test reading epochs for resting state"""
    annots = hcp.read_annot(data_type="rest", run_index=run_index, **hcp_params)

    epochs = hcp.read_epochs(data_type="rest", run_index=run_index, **hcp_params)

    _epochs_basic_checks(epochs, annots, data_type="rest")


@task_types
@run_indices_2
def test_read_epochs_task(hcp_params, data_type, run_index):
    """Test reading epochs for task"""
    annots = hcp.read_annot(
        data_type=data_type, run_index=run_index, **hcp_params
    )

    epochs = hcp.read_epochs(
        data_type=data_type, run_index=run_index, **hcp_params
    )

    _epochs_basic_checks(epochs, annots, data_type=data_type)


def _check_bounds(array, bounds, atol=0.01):
    """helper for bounds checking"""
    return np.allclose(np.min(array), min(bounds), atol=atol) and np.allclose(
        np.max(array), max(bounds), atol=atol
    )


@task_types
def test_read_evoked(hcp_params, data_type):
    """Test reading evokeds."""
    all_annots = list()
    for run_index in run_inds[:2]:
        annots = hcp.read_annot(
            data_type=data_type, run_index=run_index, **hcp_params
        )
        all_annots.append(annots)

    evokeds = hcp.read_evokeds(data_type=data_type, kind="average", **hcp_params)

    n_average = sum(ee.kind == "average" for ee in evokeds)
    assert_equal(n_average, len(evokeds))

    n_chans = 248
    if data_type == "task_motor":
        n_chans += 4
    n_chans -= len(set(sum([an["channels"]["all"] for an in all_annots], [])))
    assert_equal(n_chans, len(evokeds[0].ch_names))
    assert _check_bounds(evokeds[0].times, epochs_bounds[data_type])


@task_types
@run_indices_2
def test_read_info(tmp_path, hcp_params, data_type, run_index):
    """Test reading info."""
    # with pdf file
    info = hcp.read_info(data_type=data_type, run_index=run_index, **hcp_params)
    assert_equal(
        {k for k in info["ch_names"] if k.startswith("A")}, bti_chans
    )
    # without pdf file
    # in this case the hcp code guesses certain channel labels
    cp_paths = hcp.io.file_mapping.get_file_paths(
        subject=hcp_params["subject"],
        data_type=data_type,
        run_index=run_index,
        output="raw",
        hcp_path="",
    )
    for pp in cp_paths:
        if "c," in pp:  # don't copy pdf
            continue
        (tmp_path / op.dirname(pp)).mkdir(parents=True, exist_ok=True)
        shutil.copy(hcp_params["hcp_path"] / pp, tmp_path / pp)

    info2 = hcp.read_info(
        subject=hcp_params["subject"],
        data_type=data_type,
        hcp_path=tmp_path,
        run_index=run_index,
    )
    assert len(info["chs"]) != len(info2["chs"])
    common_chs = [ch for ch in info2["ch_names"] if ch in info["ch_names"]]
    assert_equal(len(common_chs), len(info["chs"]))
    info2 = _hcp_pick_info(info2, common_chs)
    assert_equal(info["ch_names"], info2["ch_names"])
    for ch1, ch2 in zip(info["chs"], info2["chs"]):
        assert_array_equal(ch1["loc"], ch2["loc"])


@task_types
@run_indices_2
def test_read_trial_info(hcp_params, data_type, run_index):
    """Test reading trial info basics."""
    trial_info = hcp.read_trial_info(
        data_type=data_type, run_index=run_index, **hcp_params
    )
    assert "stim" in trial_info
    assert "resp" in trial_info
    assert_equal(2, len(trial_info))
    for val in trial_info.values():
        assert_array_equal(np.ndim(val["comments"]), 1)
        assert_array_equal(np.ndim(val["codes"]), 2)
