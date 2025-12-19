import os

from catboost import CatBoostClassifier
import joblib
import pandas as pd


class ModelTrainer:
    def __init__(self, data: pd.DataFrame, cfg: dict):
        """
        Initialization function of the Preprocessing class.
        
        Arguments
        ----------
        data : pd.DataFrame
            Raw data used for preprocessing.
        cfg : dict
            Preprocessing configuration.
        """
        self.data = data.copy()
        self.cfg = cfg

    def train(self):
        """
        Function that train and save the model as a joblib's artefact.
        """
        # Columns configuration
        cols = self.cfg['Columns']

        # Model hyperparameters configuration
        params = self.cfg['Hyperparameters']

        # Input data
        X = self.data.drop(columns=[cols['Exclude'],cols['Target']])
        Y = self.data[cols['Target']]

        # Training
        model = CatBoostClassifier(**params)
        model.fit(X,Y,verbose=False)

        # Saving the model
        save_config = self.cfg['Model_save']
        joblib.dump(model, os.path.join(save_config['Root_dir'], save_config['Model_name']))
