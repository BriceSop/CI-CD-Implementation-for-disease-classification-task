import dagshub
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


class ModelEvaluation:
    def __init__(self, data: pd.DataFrame, model, cfg: dict):
        """
        Initialization function of the Preprocessing class.
        
        Arguments
        ----------
        train_data : pd.DataFrame
            Train set used for evaluation.
        valid_data : pd.DataFrame
            Validation set used for evaluation.
        cfg : dict
            Preprocessing configuration.
        """
        self.data = data
        self.model = model
        self.cfg = cfg

    def predict(self, final_pred: bool = False) -> np.array:
        """
        Initialization function of the Preprocessing class.
        
        Arguments
        ----------
        final_pred : bool
            Indicates if we use the test set or not.
        
        Returns
        -------
        self: ModelEvaluation
            The same object, with predictions added.
        """
        # Columns configuration
        cols = self.cfg['Columns']

        # Preparing the data
        if not final_pred:
            self.X = self.data.drop(columns=[cols['Exclude'],cols['Target']])
            self.Y_real = self.data[cols['Target']]
        else:
            self.X = self.data.drop(columns=[cols['Exclude']])

        # Predict
        self.Y_pred = self.model.predict_proba(self.X)

        return self

    def eval_metric(self) -> int:
        """
        Compute the evaluation metric.

        Returns
        -------
        self: ModelEvaluation
            The same object, with evaluation metric added.
        """
        # Evaluation metric calculation
        self.metric = log_loss(self.Y_real, self.Y_pred)
        print(f"Evaluation metric: {self.metric}")

        return self

    def mlflow_tracking(self, model_registry: bool = False):
        """
            MLflow experiment tracking using a dagshub repo.
            
            Arguments
            ---------
            model_registry: bool
                Indicates whether to register the model or not.
            """
        # Initialize experiment
        mlflow.set_experiment("disease-classification")
        
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(self.model.get_params())

            # Log metric
            mlflow.log_metric('Log Loss', self.metric)

            # Log model signature
            signature = infer_signature(self.X, self.Y_pred)

            # Model logging and registry
            if not model_registry:
                mlflow.catboost.log_model(self.model, 
                                          name="model", 
                                          signature=signature)
            else:
                mlflow.catboost.log_model(self.model, 
                                          name="model", 
                                          registered_model_name="Catboost", 
                                          signature=signature)

