from bignmf.datasets.datasets import Datasets
import pytest

valid_datasets = {"SimulatedX2",
                  "SimulatedX3",
                  "SimulatedX1"
                 }

def test_list_all(capsys):
    Datasets.list_all() 
    captured = capsys.readouterr()
    captured = captured.out.split('\n')
    captured.pop()
    captured = set(captured)
    assert captured == valid_datasets

def test_invalid_read():
    with pytest.raises(FileNotFoundError):
        Datasets.read('x')

@pytest.mark.parametrize("dataset_name", valid_datasets)
def test_valid_read(dataset_name):
    data = Datasets.read(dataset_name)
    assert data.__class__.__name__ == 'DataFrame'