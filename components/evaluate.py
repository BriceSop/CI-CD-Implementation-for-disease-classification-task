import dagshub
import mlflow
from mlflow.models import infer_signature
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

    def predict(self):
        """
        Initialization function of the Preprocessing class.
        
        Arguments
        ----------
        data : pd.DataFrame
            Data used to predict.
        """
        # Columns configuration
        cols = self.cfg['Columns']

        # Preparing the data
        self.X = self.data.drop(columns=[cols['Exclude'],cols['Target']])
        self.Y_real = self.data[cols['Target']]

        # Predict
        self.Y_pred = self.model.predict_proba(self.X)

    def eval_metric(self):
        """
        Compute the evaluation metric.

        Arguments
        ---------
        data_set : str
            Set of data used.

        Returns
        -------
        metric:
            Evaluation metric.
        """
        # Evaluation metric calculation
        self.metric = log_loss(self.Y_real, self.Y_pred)
        print(f"Evaluation metric: {self.metric}")

        return self.metric

    def mlflow_tracking(self, model_registry: bool = False):
        """
            MLflow experiment tracking using a dagshub repo.
            
            Arguments
            ---------
            model_registry: bool
                Indicates whether to register the model or not.
            """
        # Initialize dagshub repository
        dagshub.init(repo_owner='BriceSop', 
                     repo_name='CI-CD-Implementation-for-disease-classification-task', 
                     mlflow=True)
        
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

