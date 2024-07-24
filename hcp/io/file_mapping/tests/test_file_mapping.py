import pytest
from numpy.testing import assert_equal

from hcp.io.file_mapping import get_file_paths
from hcp.io.file_mapping.file_mapping import run_map


@pytest.mark.parametrize("data_type", [
    "rest",
    "task_working_memory",
    "task_story_math",
    "task_motor",
    "noise_empty_room",
])
@pytest.mark.parametrize("run_index", range(3))
@pytest.mark.parametrize("output", [
    "raw", "epochs", "ica", "evoked", "trial_info", "bads",
])
def test_basic_file_mapping(hcp_params, data_type, run_index, output):
    """Test construction of file paths and names"""

    pytest.raises(
        ValueError,
        get_file_paths,
        data_type="sushi",
        output="raw",
        run_index=0,
        **hcp_params,
    )

    pytest.raises(
        ValueError,
        get_file_paths,
        data_type="rest",
        output="kimchi",
        run_index=0,
        **hcp_params,
    )

    # check too many runs
    if run_index >= len(run_map[data_type]):
        pytest.raises(
            ValueError,
            get_file_paths,
            data_type=data_type,
            output=output,
            run_index=run_index,
            **hcp_params,
        )
    # check no event related outputs
    elif data_type in (
        "rest",
        "noise_subject",
        "noise_empty_room",
    ) and output in ("trial_info", "evoked"):
        pytest.raises(
            ValueError,
            get_file_paths,
            data_type=data_type,
            output=output,
            run_index=run_index,
            **hcp_params,
        )
    # check no preprocessing
    elif data_type in ("noise_subject", "noise_empty_room") and output in (
        "epochs",
        "evoked",
        "ica",
        "annot",
    ):
        pytest.raises(
            ValueError,
            get_file_paths,
            data_type=data_type,
            output=output,
            run_index=run_index,
            **hcp_params,
        )
    else:
        file_names = get_file_paths(
            data_type=data_type,
            output=output,
            run_index=run_index,
            **hcp_params,
        )
        if output == "raw":
            assert_equal(sum("config" in fn for fn in file_names), 1)
            assert_equal(sum("c,rfDC" in fn for fn in file_names), 1)
