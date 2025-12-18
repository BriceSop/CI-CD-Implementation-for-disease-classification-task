import yaml


def load_yaml(filepath):
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    
    return config

if __name__ == "__main__":
    filepath = "configs/prepare.yaml"
    config = load_yaml(filepath)
    print(config)
    for key, val in config["Cat_treatment"].items():
        print(key)
        print(val)
    
    config["Cat_treatment"].keys()