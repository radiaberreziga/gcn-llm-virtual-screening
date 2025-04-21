import json

def load_model_params(path):
    with open(path, 'r') as f:
        return json.load(f)

