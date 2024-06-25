import os

import yaml


def get_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        cont = f.read()
        x = yaml.safe_load(cont)
    return x


const = get_yaml(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'constant.yaml'))
