from components.evaluate import ModelEvaluation
from utils.common import load_data_from_csv, load_model, load_yaml

# Loading Evaluation step configuration
config = load_yaml("configs/evaluate.yaml")
print(config.keys())

# Loading validation set
valid_data = load_data_from_csv(config['Validation_Data'])
valid_data.shape

# Model loading
model = load_model(config['Model'])

# Validation set's evaluation
valid_eval = ModelEvaluation(valid_data, model, config)
valid_eval.predict()
valid_eval.eval_metric()
valid_eval.mlflow_tracking(model_registry=False)