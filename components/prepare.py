import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Preprocessing:
    """
    Preprocessing pipeline.

    All transformations are applied in place.
    Each method mutates self.data, except for the last one, and returns self to allow chaining.
    """
    def __init__(self, data: pd.DataFrame, cfg: dict):
        """
        Initialization function of the Preprocessing class.
        
        Arguments
        ----------
        data : pd.DataFrame
            Raw data to preprocess.
        cfg : dict
            Preprocessing configuration.
        """
        self.data = data.copy()
        self.cfg = cfg

    def cat_aberrant_treatment(self, test_set: bool = False) -> "Preprocessing":
        """
        Function that converts aberrant values into NaN, for categorical columns.

        Arguments
        ---------
        test_set : bool
            Specify whether the data treated is the test set or not.
        
        Returns
        -------
        self : Preprocessing
            The same object, with self.data updated.
        """
        # Remove Status' treatment for test set
        if test_set:
            self.cfg['Cat_treatment'].pop('Status')

        for key, values in self.cfg['Cat_treatment'].items():
            # Collect all the modalities of the column 
            modalities = self.data[key].dropna().unique().tolist()

            # Checking if there is abnormal values
            diff = list(set(modalities) - set(values))

            if len(diff) > 0:
                # Replacement
                self.data[key] = self.data[key].replace(diff,np.nan)
        
        return self

    def format_change(self) -> "Preprocessing":
        """
        Changing the format of certain columns.
        
        Returns
        -------
        self : Preprocessing
            The same object, with self.data updated.
        """
        for key, val in self.cfg['Format_change'].items():
            # Changing to a yearly format
            self.data[val] = self.data[key]/365

            # Rounding
            self.data[val] = self.data[val].round(decimals=2)

        return self
    
    def num_aberrant_treatment(self) -> "Preprocessing":
        """
        Function that converts aberrant values into NaN, for numerical columns.
        
        Returns
        -------
        self : Preprocessing
            The same object, with self.data updated.
        """
        for key, val in self.cfg['Num_treatment'].items():
            # Treatment for N_years equal to the Age
            if key == 'N_years':
                self.data[key] = np.where(self.data[key] >= self.data[val], np.nan, self.data[key])
            
            # Treatment for the other numerical columns
            else:
                self.data[key] = np.where(self.data[key] < val[0], np.nan, self.data[key])
                self.data[key] = np.where(self.data[key] > val[1], np.nan, self.data[key])

        return self
    
    def _compute_bounds(self, col) -> int:
        """
        Compute IQR-based lower and upper bounds for a numeric column.

        Arguments
        ---------
        col: str
            Column to use for calculation.
        
        Return
        ------
            Outliers' bounds for the column.
        """
        Q1 = self.data[col].quantile(0.25)
        Q3 = self.data[col].quantile(0.75)
        IQR = Q3 - Q1
        return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    
    def outliers_treatment(self) -> "Preprocessing":
        """
        Function that replace outliers by max or min (winsoriztion).
        
        Returns
        -------
        self : Preprocessing
            The same object, with self.data updated.
        """
        # Initialize Columns Names
        num_cols = self.cfg['Columns']

        for col in num_cols['Num_cols']:
            # Computing Outliers' bounds
            low, high = self._compute_bounds(col)

            # Replacing outliers by NaN
            self.data[col] = np.where(self.data[col] < low, low, self.data[col])
            self.data[col] = np.where(self.data[col] > high, high, self.data[col])
        
        return self
    
    def cat_nan_treatment(self, test_set: bool = False) -> "Preprocessing":
        """
        Function that replace NaN by a new class named Unknown, for categorical columns.

        Arguments
        ---------
        test_set : bool
            Specify whether the data treated is the test set or not.
        
        Returns
        -------
        self : Preprocessing
            The same object, with self.data updated.
        """
        # Initialize Columns Names
        cat_cols = self.cfg['Columns']

        # Pattern for replacement
        cat_na = {}

        # Remove Status' treatment for test set
        if test_set:
            cat_cols['Cat_cols'].remove('Status')

        # Preparing the replacement pattern dictionary
        for col in cat_cols['Cat_cols']:
            cat_na[col] = "Unknown"

        # Replacing NaN by Unknown
        self.data[cat_cols['Cat_cols']] = self.data[cat_cols['Cat_cols']].fillna(value=cat_na)
        
        return self
    
    def data_splitting(self) -> pd.DataFrame:
        """
        Splitting the data into train and validation sets.
        
        Returns
        -------
        train : pd.DataFrame
            Preprocessed itermediate train dataframe.
        validation : pd.DataFrame
            Preprocessed itermediate test dataframe.
        """
        # Splitting configuration
        split_cfg = self.cfg['Splitting']

        # Split
        train, validation = train_test_split(self.data,
                                             test_size=split_cfg['Validation_size'],
                                             random_state=split_cfg['Seed']
                                            )
        
        return train, validation 