import json
import time
import random
import dgl
import ast
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from catboost import CatBoostClassifier, CatBoostRegressor

from datasets import load_dataset

def construct_graph2(data, file_name):
    G = nx.Graph()

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    nodes = list(data.index)

    G.add_nodes_from(nodes)
    simularity = cosine_similarity(data.values, data.values)
    
    # sorted_simul = []
    for node in nodes:
        # reversed argsort
        sorted_simul_arg = np.argsort(simularity[node])[::-1]
        # exclude itself
        top5 = sorted_simul_arg[1:6]
        G.add_edges_from([node, n] for n in top5)
    
    nx.write_graphml(G, file_name)


def feature_vector(X, y, path):
    model = CatBoostClassifier(iterations=100,
                                  depth=6,
                                  learning_rate=0.1,
                                  loss_function='MultiClass',
                                  random_seed=0,
                                  nan_mode='Min',
                                  allow_const_label=True)
    
    # extract train masks
    with open(f'{path}/masks.json') as f:
        masks = json.load(f)
    train_masks = masks['0']['train']
    
    X_train = X.iloc[train_masks]
    y_train = y.iloc[train_masks]
    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", y_train.shape)

    model.fit(X_train, y_train, verbose=False)
    prediction = model.predict_proba(X)
    # pred = model.predict(X)
    # print((y == pred.max(1)).sum().item()/y.shape[0])
    leaf_index = model.calc_leaf_indexes(X)
    return prediction, leaf_index


def gen_masks(data, path):
    masks = {str(i):{"train":[], "val":[], "test":[]} for i in range(5)}
    scale_config = [int(0.6*len(data)), int(0.2*len(data)), int(0.2*len(data))]
    # print(scale_config)

    for key in list(masks.keys()):
        ids = list(data.index)
        random.shuffle(ids)

        masks[key]["train"] = ids[:scale_config[0]]
        masks[key]["val"] = ids[scale_config[0]:scale_config[0]+scale_config[1]]
        masks[key]["test"] = ids[scale_config[0]+scale_config[1]:]

    write_json(masks, path)

def sampling_masks(data, label, path, num):
    ids = list(data.index)
    masks = {str(i):{"train":[], "val":[], "test":[]} for i in range(5)}
    classes = int(max(label['class'])) + 1
    class_dict = {k:[] for k in range(classes)}
    [class_dict[int(label.iloc[i])].append(i) for i in ids]

    for key in list(masks.keys()):
        # shuffle ids
        [random.shuffle(class_dict[i]) for i in range(classes)]

        masks[key]["train"] = [item for i in range(classes) for item in class_dict[i][:num]]
        masks[key]["val"] = [item for i in range(classes) for item in class_dict[i][num:2*num]]
        masks[key]["test"] = [item for i in range(classes) for item in class_dict[i][2*num:]]

        # random shuffle again
        random.shuffle(masks[key]["train"])
        random.shuffle(masks[key]["val"])
        random.shuffle(masks[key]["test"])
    
    write_json(masks, path)
        

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


# processing huggingface small datasets
def data_process(dataset):
    for table_name, table, task in zip(dataset['dataset_name'], dataset['table'], dataset['task']):
        table = ast.literal_eval(table)
        if task == 'binclass':
            data[table_name] = {
                'X_num': None if not table['X_num'] else pd.DataFrame.from_dict(table['X_num']),
                'X_cat': None if not table['X_cat'] else pd.DataFrame.from_dict(table['X_cat']),
                'y': np.array(table['y']),
                'y_info': table['y_info'],
                'task': task,
            }
    return data

# path = 'pol'
# path20 = path + '_s20'
# path10 = path + '_s10'
# path4 = path + '_s4'

# data = pd.read_csv(f'{path}/X.csv')
# label = pd.read_csv(f'{path}/y.csv')

# # label.to_csv("y.csv", index=False, header=["class"])

# # create masks.json
# print("generating masks")
# gen_masks(data, f"{path}/masks.json")
# sampling_masks(data, label, path=f'{path10}/masks.json', num=10)  # s10
# sampling_masks(data, label, path=f'{path20}/masks.json', num=20)  # s20
# sampling_masks(data, label, path=f'{path4}/masks.json', num=4)  # s4
# print("masks.json finished")

# path1 = [path, path20, path10, path4]

# # feature simul graph
# construct_graph2(data, "graph.graphml")
# print("graph.graphml Finished")

# for path in path1:
#     print("\npath:", path)
#     start_time = time.time()

#     # prediction simul graph
#     prediction, leaf_index = feature_vector(data, label, path)
#     construct_graph2(prediction, f'{path}/pred_graph.graphml')
#     print("pred_graph.graphml Finished")

#     # leaf index simul graph
#     construct_graph2(leaf_index, f'{path}/leaf_graph.graphml')
#     print("leaf_graph.graphml Finished")
#     print("Time comsuming:", time.time()-start_time)


data = {}
datasets = load_dataset('jyansir/excelformer')
train_data, val_data, test_data = datasets['train'].to_dict(), datasets['val'].to_dict(), datasets['test'].to_dict()
train_data = data_process(train_data)
val_data = data_process(val_data)
test_data = data_process(test_data)

# for key in train_data.keys():
train = train_data['[kaggle]Analytics Vidhya Loan Prediction']
val = val_data['[kaggle]Analytics Vidhya Loan Prediction']
test = test_data['[kaggle]Analytics Vidhya Loan Prediction']
combined_X_num = pd.concat([train['X_num'], val['X_num'], test['X_num']], axis=0, ignore_index=True)
combined_X_cat = pd.concat([train['X_cat'], val['X_cat'], test['X_cat']], axis=0, ignore_index=True)
combined_X = pd.concat([combined_X_cat, combined_X_num], axis=1)
combined_y = np.append(np.append(train['y'], val['y']), test['y'])
combined_y = pd.DataFrame(combined_y, columns=['class'])

print(combined_X)
print(combined_y)

# print(train_data.keys())

# path = 
# path20 = path + '_s20'
# path10 = path + '_s10'
# path4 = path + '_s4'

# data = pd.read_csv(f'{path}/X.csv')
# label = pd.read_csv(f'{path}/y.csv')

# # label.to_csv("y.csv", index=False, header=["class"])

# # create masks.json
# print("generating masks")
# gen_masks(data, f"{path}/masks.json")
# sampling_masks(data, label, path=f'{path10}/masks.json', num=10)  # s10
# sampling_masks(data, label, path=f'{path20}/masks.json', num=20)  # s20
# sampling_masks(data, label, path=f'{path4}/masks.json', num=4)  # s4
# print("masks.json finished")

# path1 = [path, path20, path10, path4]

# # feature simul graph
# construct_graph2(data, "graph.graphml")
# print("graph.graphml Finished")

# for path in path1:
#     print("\npath:", path)
#     start_time = time.time()

#     # prediction simul graph
#     prediction, leaf_index = feature_vector(data, label, path)
#     construct_graph2(prediction, f'{path}/pred_graph.graphml')
#     print("pred_graph.graphml Finished")

#     # leaf index simul graph
#     construct_graph2(leaf_index, f'{path}/leaf_graph.graphml')
#     print("leaf_graph.graphml Finished")
#     print("Time comsuming:", time.time()-start_time)