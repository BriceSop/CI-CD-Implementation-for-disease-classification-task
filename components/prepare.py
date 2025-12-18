import numpy as np
import pandas as pd


class Preprocessing:
    def __init__(self,data:pd.DataFrame,cfg:dict):
        """
        Initialization function of the Preprocessing class.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw data used for preprocessing.
        cfg : dict
            Preprocessing configuration.
        """
        self.data = data.copy()
        self.cfg = cfg

    def cat_aberrant_treatment(self) -> pd.DataFrame:
        """
        Function that converts aberrant values into NaN, for categorical columns.
        
        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe.
        """
        for key, val in self.cfg['Cat_treatment'].items():
            # Preparing nan values used to replace aberrant values
            nan_val = [np.nan]*len(val)

            # Pattern dict
            pattern = dict(zip(val,nan_val))

            # Replacement
            self.data[key] = self.data[key].replace(pattern)
        
        return self.data

    def format_change(self) -> pd.DataFrame:
        """
        Changing the format of certain columns.
        
        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe.
        """
        for key, val in self.cfg['Format_change'].items():
            # Changing to a yearly format
            self.data[key] = self.data[key]/365

            # Rounding
            self.data[key] = self.data[key].round(decimals=2)

            # Rename according to new format
            self.data.rename(columns={key:val},inplace=True)

        return self.data
    
    def num_aberrant_treatment(self) -> pd.DataFrame:
        """
        Function that converts aberrant values into NaN, for numerical columns.
        
        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe.
        """
        for key, val in self.cfg["Num_treatment"]:
            # Treatment for N_years equal to the Age
            if key == 'N_years':
                self.data[key] = np.where(self.data[key] >= self.data[val], np.nan, self.data[key])
            
            # Treatment for the other numerical columns
            else:
                self.data[key] = np.where(self.data[key] > val, np.nan, self.data[key])

        return self.data
    
    def outliers_treatment(self) -> pd.DataFrame:
        """
        Function that replace outliers by NaN.
        
        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe.
        """
        # Initialize Columns Names
        num_cols = self.cfg['Columns']
        for col in num_cols['Num_cols']:

            # Calculating first and third quartile
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)

            # Interquartile Range
            IQR = Q3 - Q1

            # Outliers' bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Replacing outliers by NaN
            self.data[col] = np.where(self.data[col] < lower_bound, np.nan, self.data[col])
            self.data[col] = np.where(self.data[col] > upper_bound, np.nan, self.data[col])
        
        return self.data
    
    def cat_nan_treatment(self) -> pd.DataFrame:
        """
        Function that replace NaN by a new class named Unknown, for categorical columns.
        
        Returns
        -------
        pd.DataFrame
            Preprocessed dataframe.
        """
        # Initialize Columns Names
        cat_cols = self.cfg['Columns']

        # Pattern for replacement
        cat_na = {}

        # Preparing the replacement pattern dictionary
        for col in cat_cols['Cat_cols']:
            cat_na[col] = "Unknown"

        # Replacing NaN by Unknown
        self.data = self.data.fillna(value=cat_na)
        
        return self.data