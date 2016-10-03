from hcp.io.file_mapping import get_file_paths
from hcp.io.file_mapping.file_mapping import run_map
import hcp.tests.config as tconf
from nose.tools import assert_raises, assert_equal


def test_basic_file_mapping():
    """Test construction of file paths and names"""

    assert_raises(ValueError,  get_file_paths,
                  subject=tconf.subject, data_type='sushi',
                  output='raw', run_index=0, hcp_path=tconf.hcp_path)

    assert_raises(ValueError,  get_file_paths,
                  subject=tconf.subject, data_type='rest',
                  output='kimchi', run_index=0,
                  hcp_path=tconf.hcp_path)

    for run_index in range(3):
        for output in tconf.hcp_outputs:
            for data_type in tconf.hcp_data_types:
                # check too many runs
                if run_index >= len(run_map[data_type]):
                    assert_raises(
                        ValueError,  get_file_paths,
                        subject=tconf.subject, data_type=data_type,
                        output=output, run_index=run_index,
                        hcp_path=tconf.hcp_path)
                # check no event related outputs
                elif (data_type in ('rest', 'noise_subject',
                                    'noise_empty_room') and
                        output in ('trial_info', 'evoked')):
                    assert_raises(
                        ValueError,  get_file_paths,
                        subject=tconf.subject, data_type=data_type,
                        output=output, run_index=run_index,
                        hcp_path=tconf.hcp_path)
                # check no preprocessing
                elif (data_type in ('noise_subject',
                                    'noise_empty_room') and output in
                        ('epochs', 'evoked', 'ica', 'annot')):
                    assert_raises(
                        ValueError,  get_file_paths,
                        subject=tconf.subject, data_type=data_type,
                        output=output, run_index=run_index,
                        hcp_path=tconf.hcp_path)
                else:
                    file_names = get_file_paths(
                        subject=tconf.subject, data_type=data_type,
                        output=output, run_index=run_index,
                        hcp_path=tconf.hcp_path)
                    if output == 'raw':
                        assert_equal(
                            sum('config' in fn for fn in file_names), 1)
                        assert_equal(
                            sum('c,rfDC' in fn for fn in file_names), 1)
