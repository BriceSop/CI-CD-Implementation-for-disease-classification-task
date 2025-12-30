import os

import joblib
import pandas as pd
from utils.common import (
    create_directory,
    load_data_from_csv,
    load_model,
    load_yaml,
    save_data_to_csv,
)


def test_load_yaml():
    cfg = load_yaml("configs/prepare.yaml")
    assert cfg is not None
    assert type(cfg) is dict

def test_load_data_from_csv():
    data = load_data_from_csv("data/raw/train.csv")
    assert type(data) is pd.DataFrame
    assert data.empty is False

def test_save_data_to_csv(tmp_path):
    data = pd.DataFrame({"Test":[1,2,3]})
    filename = 'dummy.csv'
    save_data_to_csv(data, tmp_path, filename)
    filepath = os.path.join(tmp_path, filename)
    assert os.path.exists(filepath)

def test_create_directory(tmp_path):
    create_directory(tmp_path)
    assert os.path.exists(tmp_path)

def test_load_model(tmp_path):
    data = pd.DataFrame({"Test":[1,2,3]})
    filepath = tmp_path / 'dummy.joblib'
    joblib.dump(data,filepath)
    test = load_model(filepath)
    assert type(test) is pd.DataFrame