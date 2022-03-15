from bignmf.datasets.datasets import Datasets
from bignmf.models.snmf.standard import StandardNmf
import pytest
import pandas as pd
import numpy as np

valid_datasets = ["SimulatedX1", "SimulatedX2"]


class TestInvalidParams:
    "Test invalid input parameters to the class initialisations"

    @pytest.mark.parametrize(
        "invalid_dataset, expected_error_msg",
        [
            (3, "The given dataset is not a DataFrame"),
            (pd.DataFrame(-np.eye(3)), "The input matrix must be non-negative"),
        ],
    )
    def test_invalid_dataset(self, invalid_dataset, expected_error_msg):
        "Test invalid datasets and check the error messages"
        rank = 3
        with pytest.raises(ValueError) as excinfo:
            model = StandardNmf(invalid_dataset, rank)
        error_msg = excinfo.value.args[0]
        assert error_msg == expected_error_msg

    @pytest.mark.parametrize("rank", [1, 3.5])
    def test_invalid_ranks(self, rank):
        valid_data = Datasets.read(valid_datasets[0])
        with pytest.raises(ValueError) as excinfo:
            model = StandardNmf(valid_data, rank)
        expected_error_msg = (
            "The given rank is invalid. Choose an integral rank greater than 1."
        )
        error_msg = excinfo.value.args[0]
        assert error_msg == expected_error_msg


@pytest.mark.parametrize("valid_dataset_name", valid_datasets)
def test_valid_params(valid_dataset_name):
    data = Datasets.read(valid_dataset_name)
    rank = 3
    model = StandardNmf(data, rank)
    assert np.array_equal(model.x, data.values)
    assert model.k == rank
    assert model.row_index == list(data.index)
    assert model.column_index == list(data)
