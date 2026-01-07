import os

import joblib
import pandas as pd
import pytest
from utils.common import (
    create_directory,
    load_data_from_csv,
    load_model,
    load_yaml,
    save_data_to_csv,
)


# Fixture and csv data generation
@pytest.fixture(scope="session")
def csv_data(tmp_path_factory):
    data = pd.DataFrame({"Test":[1,2,3]})
    fn = tmp_path_factory.mktemp("data") / "tmp_data.csv"
    data.to_csv(fn, index=False)
    return fn

def test_load_yaml():
    # GIVEN a yaml file's path
    # WHEN load_yaml is applied
    cfg = load_yaml("configs/prepare.yaml")
    # THEN cfg is not empty and should be a dict
    assert cfg is not None
    assert type(cfg) is dict

def test_load_data_from_csv(csv_data):
    # GIVEN a csv file's path
    # WHEN load_data_from_csv is applied
    data = load_data_from_csv(csv_data)
    # THEN data should be a not empty DataFrame
    assert type(data) is pd.DataFrame
    assert data.empty is False

def test_save_data_to_csv(tmp_path):
    # GIVEN a data to save and a filepath
    data = pd.DataFrame({"Test":[1,2,3]})
    filename = 'dummy.csv'
    # WHEN save_data_to_csv is applied
    save_data_to_csv(data, tmp_path, filename)
    filepath = os.path.join(tmp_path, filename)
    # THEN filepath should exist
    assert os.path.exists(filepath)

def test_create_directory(tmp_path):
    # GIVEN a directory path
    # WHEN create_directory is applied
    create_directory(tmp_path)
    # THEN the new directory should exist
    assert os.path.exists(tmp_path)

def test_load_model(tmp_path):
    # GIVEN a filepath
    # WHEN load_model is applied
    data = pd.DataFrame({"Test":[1,2,3]})
    filepath = tmp_path / 'dummy.joblib'
    joblib.dump(data,filepath)
    test = load_model(filepath)
    # THEN test type should not be None
    assert type(test) is pd.DataFrame