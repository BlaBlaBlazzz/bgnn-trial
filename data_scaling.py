import json
import numpy as np
import random
import pandas as pd


def read_json(path):
    with open(path) as f:
        data = json.load(f)    
    return data

def mask_scaling(data, config):
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

def write_sampling_masks():
    data_path = "./datasets/wdbc/y.csv"
    mask_path = './datasets/wdbc/masks.json'
    label = pd.read_csv(data_path)
    masks = read_json(mask_path)
    # sample
    sampling(label, masks)
    write_json(masks, "masks.json")

def sampling(label, masks):
    for key in masks:
        ids = masks[key]["train"] + masks[key]["val"] + masks[key]["test"]
        random.shuffle(ids)
        # initialize class dictionary
        class_dict = {k:[] for k in range(int(max(label["class"])) + 1)}
        [class_dict[int(label.iloc[i])].append(i) for i in ids]

        # sampling 10 ids in train, val progress
        masks[key]["train"] = [item for i in range(int(max(label["class"])) + 1) for item in class_dict[i][:4]]
        masks[key]["val"] = [item for i in range(int(max(label["class"])) + 1) for item in class_dict[i][4:8]]
        masks[key]["test"] = [item for i in range(int(max(label["class"])) + 1) for item in class_dict[i][8:]]

        # random shuffle
        random.shuffle(masks[key]["train"])
        random.shuffle(masks[key]["val"])
        random.shuffle(masks[key]["test"])
    # print(masks)

write_sampling_masks()        

