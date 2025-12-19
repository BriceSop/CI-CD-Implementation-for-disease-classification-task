from components.prepare import Preprocessing
from utils.common import create_directory, load_data_from_csv, load_yaml, save_data_to_csv

config = load_yaml("configs/prepare.yaml")
print(config.keys())
data = load_data_from_csv("data/raw/train.csv")
data.head()

prep = Preprocessing(data, config)
prep.cat_aberrant_treatment()
prep.format_change()
prep.num_aberrant_treatment()
prep.outliers_treatment()
prep.cat_nan_treatment()
train, validation = prep.data_splitting()

dir_path = config['Splitting']['Dir_path']
train_file_name = config['Splitting']['Train_file']
validation_file_name = config['Splitting']['Validation_file']
create_directory(dir_path)
save_data_to_csv(train, dir_path, train_file_name)
save_data_to_csv(validation, dir_path, validation_file_name)