from catboost import CatBoostRegressor, Pool
import numpy as np
import time

# 創建一個示例數據集
X = np.random.rand(50000, 5)
y = np.random.rand(50000)

# 初始化 CatBoostRegressor 模型
model = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1)
model.fit(X, y)

# 創建 Pool 對象
pool = Pool(X)

# 使用 predict 方法獲取每個樹的葉子節點索引
leaf_indexes = model.predict(pool, prediction_type='RawFormulaVal')
start = time.time()
leaf_indexes = model.calc_leaf_indexes(pool)
print(time.time() - start)
# leaf_indexes 是一個 NumPy 數組，形狀為 (num_samples, num_trees)
print(leaf_indexes.shape)
