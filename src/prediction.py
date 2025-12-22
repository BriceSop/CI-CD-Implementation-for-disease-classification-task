from components.evaluate import ModelEvaluation
from components.training import ModelTrainer
import pandas as pd
from utils.common import (
    create_directory,
    load_data_from_csv,
    load_model,
    load_yaml,
    save_data_to_csv,
)

# Loading all the needed configurations
train_config = load_yaml("configs/training.yaml")
pred_config = load_yaml("configs/prediction.yaml")

# Loading preprocessed datasets
train_data = load_data_from_csv(pred_config['Data']['Train'])
test_data = load_data_from_csv(pred_config['Data']['Test'])

# Model initialization
model = ModelTrainer(train_data, train_config)

# Model training
model.train()

# Model loading
model = load_model(pred_config['Model'])

# Model Inference
valid_eval = ModelEvaluation(test_data, model, train_config)
pred = valid_eval.predict(final_pred=True)

# Construction of the submission dataframe
final_data = pd.DataFrame(index=test_data.index)
final_data['id'] = test_data['id']
final_data[['Status C','Status CL','Status D']] = pred
final_data.head()

# Saving the submission data
dir_path = pred_config['Inference_Save']['Dir_path']
file_name = pred_config['Inference_Save']['File_name']
create_directory(dir_path)
save_data_to_csv(final_data, dir_path, file_name)
