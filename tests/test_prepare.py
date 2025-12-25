from components.prepare import Preprocessing
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
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
        "Drug": np.random.choice(["Placebo", "D-penicillamine", np.nan], size=n),
        "Sex": np.random.choice(["F", "M", np.nan], size=n),
        "Ascites": np.random.choice(["Y", "N", np.nan], size=n),
        "Hepatomegaly": np.random.choice(["Y", "N", np.nan], size=n),
        "Spiders": np.random.choice(["Y", "N", np.nan], size=n),
        "Edema": np.random.choice(["Y", "N", "S", np.nan], size=n),
        "Status": np.random.choice(["C","CL","D", np.nan], size=n),
    }
    return pd.DataFrame(data)

@pytest.fixture
def prep():
    # Test DataFrame and preprocessing configuration
    df = generate_testing_df(20, 4)
    cfg = load_yaml("configs/prepare.yaml")
    return Preprocessing(df, cfg)

@pytest.mark.parametrize("col, expected", [
    ("Drug", ['Placebo', 'D-penicillamine']), 
    ("Sex", ['F', 'M']),
    ("Ascites", ['Y', 'N']),
    ("Hepatomegaly", ['Y', 'N']),
    ("Spiders", ['Y', 'N']),
    ("Edema", ['Y', 'N', 'S']),
    ("Status", ['C', 'CL', 'D'])
])
def test_cat_aberrant_treatment(prep, col, expected):
    # GIVEN a Preprocessing object with aberrant categorical values
    # WHEN cat_aberrant_treatment is applied
    res = prep.cat_aberrant_treatment().data
    vals = res[col].dropna().unique().tolist()
    # THEN only the allowed values remain in the column
    assert set(vals) == set(expected)

@pytest.mark.parametrize("col, new_col",[
    ("Age", "Age_in_years"),
    ("N_Days","N_years")
])
def test_format_change(prep, col, new_col):
    # GIVEN a DataFrame with columns to convert
    # WHEN format_change is applied
    res = prep.format_change().data
    expected = pd.DataFrame()
    expected[new_col] = (prep.data[col] / 365).round(2)
    # THEN the columns are renamed and converted correctly
    assert_series_equal(res[new_col], expected[new_col], check_dtype=False)

@pytest.mark.parametrize("col, bounds",[
    ("Age_in_years", [16, 100]),
    ("N_years", None),
    ("Bilirubin", [0.1, 50]),
    ("Cholesterol", [60, 1500]),
    ("Albumin", [1, 6.5]),
    ("Copper", [10, 800]),
    ("Alk_Phos", [20, 6000]),
    ("SGOT", [5, 10000]),
    ("Tryglicerides", [20, 5000]),
    ("Platelets", [10, 1000]),
    ("Prothrombin", [8, 70])
])
def test_num_aberrant_treatment(prep, col, bounds):
    # GIVEN a DataFrame with numeric aberrant values
    # WHEN num_aberrant_treatment is applied
    res = prep.num_aberrant_treatment().data
    # THEN values respect the defined bounds
    if col == "N_years":
        assert (res["N_years"] < res["Age_in_years"]).all()
    else:
        col_values = res[col].dropna()
        assert col_values.min() >= bounds[0]
        assert col_values.max() <= bounds[1]

@pytest.mark.parametrize("col",[
    "Bilirubin", "Cholesterol", "Albumin", "Copper",
    "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
])
def test_outliers_treatment(prep, col):
    # GIVEN a DataFrame with outliers
    # WHEN outliers_treatment is applied
    res = prep.outliers_treatment().data
    low, high = prep._compute_bounds(col)
    col_values = res[col].dropna()
    # THEN all values are within the computed bounds
    assert col_values.min() >= low
    assert col_values.max() <= high

def test_cat_nan_treatment(prep):
    # GIVEN a DataFrame with NaN in categorical columns
    # WHEN cat_nan_treatment is applied
    res = prep.cat_nan_treatment().data
    cat_data = res[prep.cfg['Columns']['Cat_cols']]
    # THEN all NaN values are replaced with "Unknown"
    assert not cat_data.isnull().any().any()

def test_data_splitting(prep):
    # GIVEN a preprocessed DataFrame
    # WHEN data_splitting is applied
    train, valid = prep.data_splitting()
    # THEN the sum of the rows of the splits equals the DataFrame length
    assert len(train) + len(valid) == len(prep.data)
