import os

from components.training import ModelTrainer
import numpy as np
import pandas as pd
import pytest
from utils.common import load_yaml


# Fixture and test DataFrame generation
def generate_testing_df(n, seed):
    np.random.seed(seed)
    data = {
        "id": np.random.randint(0, 30000, n),
        "Age": np.random.randint(12000, 30000, n),
        "N_Days": np.random.randint(0, 4500, n),
        "Age_in_years": np.random.randint(16, 90, n),
        "N_years": np.random.randint(0, 10, n),
        "Bilirubin": np.random.uniform(0.5, 25, n),
        "Cholesterol": np.random.randint(50, 2000, n),
        "Albumin": np.random.uniform(2.5, 5.5, n),
        "Alk_Phos": np.random.uniform(50, 2000, n),
        "SGOT": np.random.uniform(10, 1500, n),
        "Tryglicerides": np.random.randint(10, 6000, n),
        "Copper": np.random.uniform(50, 400, n),
        "Platelets": np.random.randint(5, 1500, n),
        "Prothrombin": np.random.uniform(10, 35, n),
        "Stage": np.random.randint(1, 5, n),
        "Drug": np.random.choice(["Placebo", "D-penicillamine"], size=n),
        "Sex": np.random.choice(["F", "M"], size=n),
        "Ascites": np.random.choice(["Y", "N"], size=n),
        "Hepatomegaly": np.random.choice(["Y", "N"], size=n),
        "Spiders": np.random.choice(["Y", "N"], size=n),
        "Edema": np.random.choice(["Y", "N", "S"], size=n),
        "Status": np.random.choice(["C","CL","D"], size=n),
    }
    return pd.DataFrame(data)

@pytest.fixture
def prep():
    # Test DataFrame and preprocessing configuration
    df = generate_testing_df(5, 4)
    cfg = load_yaml("configs/training.yaml")
    return ModelTrainer(df, cfg)

def test_model_attr(prep):
    model = prep.train(save_model=False)
    assert model is not None
    assert hasattr(model, 'tree_count_')
    assert hasattr(model, 'random_seed_')
    assert hasattr(model, 'feature_names_')
    assert hasattr(model, 'classes_')

def test_model_seed(prep):
    model = prep.train(save_model=False)
    seed = prep.cfg["Hyperparameters"]["random_state"]
    assert model.random_seed_ == seed

def test_model_file(prep, tmp_path):
    prep.cfg['Model_save']['Root_dir'] = tmp_path
    prep.train(save_model=True)
    filepath = os.path.join(tmp_path, prep.cfg['Model_save']['Model_name'])
    assert os.path.exists(filepath)
