from catboost import CatBoostRegressor, Pool
import numpy as np
# Initialize data

train_data = [[1, 4, 5, 6],
              [4, 5, 6, 7],
              [30, 40, 50, 60]]

eval_data = [[2, 4, 6, 8],
             [1, 4, 50, 60]]

train_labels = np.array([[10, 10], [20, 20], [30, 30]])
print(train_labels.shape)
pool = Pool(train_data, train_labels, cat_features=[])
# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=2,
                          learning_rate=1,
                          depth=2,
                          loss_function = "MultiRMSE")
# Fit model
model.fit(pool, verbose=False)
# Get predictions
preds = model.predict(eval_data)
print(preds)
