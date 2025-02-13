# the code is catastrophic, i'll fix it later

import json
import os
import pandas as pd
from pathlib import Path

def aggregate_result(algos, results, path, task):
    with open(path) as f:
        result_dict = json.load(f)
    
    aggr_models = {algo:[] for algo in algos}

    for key in result_dict.keys():
        for model in algos:
            if key.startswith(model):
                aggr_models[model].append(key)
    # print(aggr_models)

    for models in aggr_models.values():
        # print(models)
        compare = min if task == 'regression' else max
        best_score = float("inf") if task == 'regression' else float('-inf')
        best_score_var = 0
        for algo in models:
            best_score = round(compare(result_dict[algo][0], best_score), 4)
            
            # if result_dict[algo][0] < best_score:
            #     best_score = round(result_dict[algo][0], 4)
            #     best_score_var = round(result_dict[algo][1], 3)
        
        if not models == []:
            results.append(best_score)
            # results_var.append(best_score_var)

    return results

def main(task):
    if task == 'classification':
        sd_path = Path(__file__).parent.parent / 'datasets' / 'huggingFace_sd'
    else:
        sd_path = Path(__file__).parent.parent / 'datasets' / 'huggingFace_sd_regression'
    datasets_ls = os.listdir(sd_path)
    print(datasets_ls)

    algos = ['bgnn-', 'bgnn_v2', 'resgnn-', 'resgnn_LI', 'resgnnXG', 'gnn-', 'unknown', 'catboost', 'lightgbm', 'xgboost', 'RandomForest',
         'ExcelFormer-MixupNone', 'ExcelFormer-Mixuphidden', 'ExcelFormer-Mixupfeature', 'trompt', 'tabnet', 
         'tabtransformer', 'fttransformer', 'aggBGNN-', 'aggBGNN_dnf', 'aggBGNN_dg', 'aggBGNN_v2']
    
    # algos = ['bgnn-', 'bgnn_v2', 'resgnn-', 'resgnn_LI', 'resgnnXG', 'unknown', 'catboost', 
    #      'ExcelFormer', 'trompt', 'tabnet', 
    #      'tabtransformer', 'fttransformer', 'aggBGNN-', 'aggBGNN_dnf', 'aggBGNN_dg', 'aggBGNN_v2']

    aggr_results = []

    for dataset in datasets_ls:
        # binary classification .6/.2/.2
        # selected_model_path = f'results/huggingFace_sd/{dataset}/{dataset}_v2/selected_models/aggregated_results.json'
        # transformer_path = f'results/huggingFace_sd/{dataset}/{dataset}/transformers/aggregated_results.json'
        # aggBGNN_path = f'results/huggingFace_sd/{dataset}/{dataset}/aggBGNN/aggregated_results.json'
        # aggBGNN_v2_path = f'results/huggingFace_sd/{dataset}/{dataset}/aggBGNN_v2/aggregated_results.json'

        # binary classification s10
        # selected_model_path = f'results/huggingFace_sd/{dataset}/{dataset}_s10/selected_models/aggregated_results.json'
        # aggBGNN_path = f'results/huggingFace_sd/{dataset}/{dataset}_s10/aggBGNN/aggregated_results.json'
        # aggBGNN_v2_path = f'results/huggingFace_sd/{dataset}/{dataset}/aggBGNN_v2/aggregated_results.json'

        # binary classification s4
        # selected_model_path = f'results/huggingFace_sd/{dataset}/{dataset}_s4/selected_models/aggregated_results.json'
        # aggBGNN_path = f'results/huggingFace_sd/{dataset}/{dataset}_s4/aggBGNN/aggregated_results.json'

        # regression .6/.2/.2
        selected_model_path = f'results/huggingFace_sd_regression/{dataset}/{dataset}/selected_models/aggregated_results.json'
        aggBGNN_path = f'results/huggingFace_sd_regression/{dataset}/{dataset}/aggBGNN/aggregated_results.json'

        # regression s10
        # selected_model_path = f'results/huggingFace_sd_regression/{dataset}/{dataset}_s10/selected_models/aggregated_results.json'
        # aggBGNN_path = f'results/huggingFace_sd_regression/{dataset}/{dataset}_s10/aggBGNN/aggregated_results.json'

        if not os.path.exists(selected_model_path):
            continue
        if not os.path.exists(aggBGNN_path):
            continue

        results = []

        results = aggregate_result(algos, results, selected_model_path, task)
        results = aggregate_result(algos, results, aggBGNN_path, task)
        # results = aggregate_result(algos, results, aggBGNN_v2_path, task)
    
        aggr_results.append(results)

    # with open("results.txt", 'a') as f:
    #     f.write(str(dataset) + "\n")
    #     for acc, var in zip(results, results_var):
    #         f.write(str(acc) + " ± " + str(var) + "\n")
    #     f.write('\n')

    df = pd.DataFrame(aggr_results)
    columns = ['bgnn', 'bgnn_v2', 'resgnn-', 'resgnn_LI', 'resgnnXG', 'gnn', 'unknown', 'catboost', 'lightgbm', 'xgboost', 'RandomForest',
         'ExcelFormer-MixupNone', 'ExcelFormer-Mixuphidden', 'ExcelFormer-Mixupfeature', 'trompt', 'tabnet', 
         'tabtransformer', 'fttransformer', 'aggBGNN-', 'aggBGNN_dnf', 'aggBGNN_dg', 'aggBGNN_v2']
    
    print(columns)
    print(df)
    ascending = True if task == 'regression' else False
    ranks = df.rank(ascending=ascending, method='min', axis=1)

    print(ranks)
    Mean = ranks.mean(axis=0)
    Std = ranks.std(axis=0)

    for mean, std in zip(list(Mean), list(Std)):
        print(round(mean, 3), "±", round(std, 3))
    

if __name__ == '__main__':
    task = 'regression'
    main(task)





    

'''

sd_path = Path(__file__).parent.parent / 'datasets' / 'huggingFace_sd'
datasets_ls = os.listdir(sd_path)
print(datasets_ls)

algos = ['bgnn-', 'bgnn_v2', 'resgnn', 'resgnn_LI', 'resgnnXG', 'unknown', 'catboost',
         'ExcelFormer', 'trompt', 'tabnet', 'tabtransformer', 'fttransformer', 'aggBGNN-', 'aggBGNN_dnf', 'aggBGNN_dg', 'aggBGNN_v2']
aggr_results = []

for dataset in datasets_ls:
    if not os.path.exists(f'results/huggingFace_sd/{dataset}/{dataset}_v2/selected_models/aggregated_results.json'):
        continue
    if not os.path.exists(f'results/huggingFace_sd/{dataset}/{dataset}/aggBGNN_v2/aggregated_results.json'):
        continue
    # print(dataset)
    results = []
    results_var = []
    with open(f'results/huggingFace_sd/{dataset}/{dataset}_v2/selected_models/aggregated_results.json') as f:
        result_dict1 = json.load(f)

    aggr_models = {algo:[] for algo in algos}

    # selected models
    for key in result_dict1.keys():
        for model in algos:
            if model in key:
                aggr_models[model].append(key)
        
    for models in aggr_models.values():
        best_score = 0
        best_score_var = 0
        for algo in models:
            if result_dict1[algo][0] >= best_score:
                best_score = round(result_dict1[algo][0], 4)
                best_score_var = round(result_dict1[algo][1], 3)
        
        if not models == []:
            results.append(best_score)
            results_var.append(best_score_var)
            # print(best_score, "±", best_score_var)
    

    # aggBGNN
    with open(f'results/huggingFace_sd/{dataset}/{dataset}/aggBGNN_v2/aggregated_results.json') as f:
        result_dict2 = json.load(f)

    aggr_models = {algo:[] for algo in algos}

    for key in result_dict2.keys():
        for model in algos:
            if model in key:
                aggr_models[model].append(key)
        
    for models in aggr_models.values():
        best_score = 0
        best_score_var = 0
        for algo in models:
            if result_dict2[algo][0] >= best_score:
                best_score = round(result_dict2[algo][0], 4)
                best_score_var = round(result_dict2[algo][1], 3)
        
        if not models == []:
            results.append(best_score)
            results_var.append(best_score_var)
            # print(best_score, "±", best_score_var)
    
    aggr_results.append(results)

    with open("results.txt", 'a') as f:
        f.write(str(dataset) + "\n")
        for acc, var in zip(results, results_var):
            f.write(str(acc) + " ± " + str(var) + "\n")
        f.write('\n')

df = pd.DataFrame(aggr_results)
print(df)
ranks = df.rank(ascending=False, method='min', axis=1)

print(ranks)
Mean = ranks.mean(axis=0)
Std = ranks.std(axis=0)

for mean, std in zip(list(Mean), list(Std)):
    print(round(mean, 3), "±", round(std, 3))


'''