from components.training import ModelTrainer
from utils.common import load_data_from_csv, load_yaml

# Loading Training step configuration
config = load_yaml("configs/training.yaml")
print(config.keys())

# Loading preprocessed training data
train_data = load_data_from_csv(config['Input_Data'])
train_data.shape

# Model initialization
model = ModelTrainer(train_data, config)

# Model training
model.train()