from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import numpy as np
from collections import defaultdict as ddict

class RandomForest:
    def __init__(self, task='regression', max_depth=6, lr=0.1, n_estimators=100, max_features=20, min_samples_leaf=1):
        self.task = task
        self.max_depth = max_depth
        self.learning_rate = lr
        self.n_estimator = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf


    def init_model(self):
        random_forest_model_obj = RandomForestRegressor if self.task == 'regression' else RandomForestClassifier
        self.model = random_forest_model_obj(n_estimators=self.n_estimator,
                                             max_depth=self.max_depth,
                                             max_features=self.max_features,
                                             min_samples_leaf=self.min_samples_leaf)

    def get_metrics(self, data, label):
        pred = self.model.predict(data)
        loss = mean_squared_error(label, pred) ** 0.5
        acc = accuracy_score(label, pred)
        return loss, acc

    def get_test_metric(self, metrics, metric_name):
        if metric_name == 'loss':
            val_epoch = np.argmin([acc[1] for acc in metrics[metric_name]])
        else:
            val_epoch = np.argmax([acc[1] for acc in metrics[metric_name]])
        min_metric = metrics[metric_name][val_epoch]
        return min_metric

    def save_metrics(self, metrics, fn):
        with open(fn, "w+") as f:
            for key, value in metrics.items():
                print(key, value, file=f)

    def train_val_test_split(self, X, y, train_mask, val_mask, test_mask):
        X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
        X_val, y_val = X.iloc[val_mask], y.iloc[val_mask]
        X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def fit(self,
            X, y, train_mask, val_mask, test_mask,
            cat_features=None, num_epochs=1000, patience=200,
            plot=False, verbose=False,
            loss_fn="", metric_name='loss'):

        encoded_X = X.copy()
        # print("X", X)
        
        X_train, y_train, X_val, y_val, X_test, y_test = \
            self.train_val_test_split(encoded_X, y, train_mask, val_mask, test_mask)
        self.init_model()

        start = time.time()
        self.model.fit(X_train, y_train)
        finish = time.time()

        # print('Finished training. Total time: {:.2f} | Number of trees: {:d} | Time per tree: {:.2f}'.format(finish - start, num_trees, (time.time() - start )/num_trees))

        # get metrics
        metrics = ddict(list)
        if self.task == 'classification':
            train_loss, train_acc = self.get_metrics(X_train, y_train)
            val_loss, val_acc = self.get_metrics(X_val, y_val)
            test_loss, test_acc = self.get_metrics(X_test, y_test)

            metrics['loss'] = [(train_loss, val_loss, test_loss)]
            metrics['accuracy'] = [(train_acc, val_acc, test_acc)]

        print('Training accuracy: {:.3f}/{:.3f}/{:.3f}'.format(*metrics['accuracy'][0]))
        return metrics

    def predict(self, X_test, y_test):
        pred = self.model.predict(X_test)

        metrics = {}
        metrics['rmse'] = mean_squared_error(pred, y_test) ** .5

        return metrics


