import json
import numpy as np
import random

def read_json(path):
    with open(path) as f:
        data = json.load(f)    
    return data

def data_scaling(data, config):
    for key in data:
        masks = []
        masks = data[key]["train"] + data[key]["val"] + data[key]["test"]
        random.shuffle(masks)
        scale_config = [int(config[0]*len(masks)), int(config[1]*len(masks)), int(config[2]*len(masks))]
        data[key]["train"] = masks[:scale_config[0]]
        data[key]["val"] = masks[scale_config[0]:scale_config[0]+scale_config[1]]
        data[key]["test"] = masks[scale_config[0]+scale_config[1]:]

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


path = './slap/masks.json'
data = read_json(path)
data_scaling(data, [0.05, 0.05, 0.9])
write_json(data, "slap_v2.json")
