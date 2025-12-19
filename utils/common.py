import os

import pandas as pd
import yaml


def load_yaml(filepath: str) -> dict:
    """
    Function that load a yaml file into dictionary.
    
    Arguments
    ----------
    filepath : str
        Path of the file.

    Raises
    ------
    e : Exception

    Returns
    -------
    config : dict
        Yaml as a dictionary.
    """
    try:
        with open(filepath, "r") as file:
            config = yaml.safe_load(file)
            print(f"yaml file: {filepath} loaded successfully")
            return config
        
    except Exception as e:
        raise e

def load_data_from_csv(filepath: str) -> pd.DataFrame:
    """
    Function used to load data from a csv file.
    
    Arguments
    ----------
    filepath : str
        Path of the file.

    Raises
    ------
    e : Exception

    Returns
    -------
    data: pd.DataFrame
        Data loaded as a Dataframe.
    """
    try:
        data = pd.read_csv(filepath, sep=',')
        print("Data loaded successfully")
        return data
    
    except Exception as e:
        raise e
    
def save_data_to_csv(data: pd.DataFrame, dir_path: str, filename: str):
    """
    Function used to save data as a csv file.
    
    Arguments
    ----------
    data : pd.DataFrame
        Data to save.
    dir_path : str
        Location of the directory used to store the csv file.
    filename : str
        Name of the file.

    Raises
    ------
    e : Exception

    Returns
    -------
    Print method
        Indicating the location of the saved data.
    """
    try:
        path = os.path.join(dir_path, filename)
        data.to_csv(path)
        return print(f"{filename} Data successfully saved at {dir_path}")
    
    except Exception as e:
        raise e

def create_directory(dir_path: str):
    """
    Function used to create a directory.
    
    Arguments
    ----------
    parent_dir : str
        Parent directory.
    dir_name : str
        Name of the new directory.

    Raises
    ------
    e : Exception

    Returns
    -------
    Print method
        Indicating the location of the created directory.
    """
    try:
        os.makedirs(dir_path,exist_ok=False)
        return print(f"Directory successfully created at {dir_path}")

    except Exception as e:
        raise e

if __name__ == "__main__":
    filepath = "configs/prepare.yaml"
    config = load_yaml(filepath)
    print(config)
    for key, val in config["Cat_treatment"].items():
        print(key)
        print(val)
    
    config["Cat_treatment"].keys()