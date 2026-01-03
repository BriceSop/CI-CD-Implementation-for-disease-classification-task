from components.prepare import Preprocessing
from utils.common import create_directory, load_data_from_csv, load_yaml, save_data_to_csv

# Loading Preprocessing step configuration
config = load_yaml("configs/prepare.yaml")
print(config.keys())

# Loading raw data
train_data = load_data_from_csv(config['Raw_Data']['Train'])
test_data = load_data_from_csv(config['Raw_Data']['Test'])
train_data.shape
test_data.shape

# Train data Preprocessing
train_prep = Preprocessing(train_data, config)
train_prep.cat_aberrant_treatment()
train_prep.format_change()
train_prep.num_aberrant_treatment()
train_prep.outliers_treatment()
train_prep.cat_nan_treatment()

# Test data Preprocessing
test_prep = Preprocessing(test_data, config)
test_prep.cat_aberrant_treatment(test_set=True)
test_prep.format_change()
test_prep.num_aberrant_treatment()
test_prep.outliers_treatment()
test_prep.cat_nan_treatment(test_set=True)

# Saving the preprocessed data
dir_path = config['Processed_Data']['Dir_path']
train_file_name = config['Processed_Data']['Train_file']
test_file_name = config['Processed_Data']['Test_file']
save_data_to_csv(train_prep.data, dir_path, train_file_name)
save_data_to_csv(test_prep.data, dir_path, test_file_name)


# Splitting Train data into train_set and validation_set
train, validation = train_prep.data_splitting()
dir_path = config['Splitting']['Dir_path']
train_file_name = config['Splitting']['Train_file']
validation_file_name = config['Splitting']['Validation_file']
create_directory(dir_path)
save_data_to_csv(train, dir_path, train_file_name)
save_data_to_csv(validation, dir_path, validation_file_name)
train.shape
validation.shape