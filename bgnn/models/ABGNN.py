import itertools
import time
import numpy as np
import torch

from catboost import Pool, CatBoostClassifier, CatBoostRegressor, sum_models
from sklearn.metrics import r2_score, accuracy_score
from .GNN import GNNModelDGL, GATDGL
from .Base import BaseModel
from sklearn import preprocessing
from tqdm import tqdm
from collections import defaultdict as ddict
import torch.nn.functional as F

class ABGNN(BaseModel):
    def __init__(self,
                 task='regression', iter_per_epoch = 10, lr=0.01, hidden_dim=64, dropout=0.,
                 only_gbdt=False, train_non_gbdt=False,
                 name='gat', trees_per_epoch = 10, use_leaderboard=False, depth=6, gbdt_lr=0.1, gbdt_alpha=1):
        super(BaseModel, self).__init__()
        self.learning_rate = lr
        self.hidden_dim = hidden_dim
        self.task = task
        self.dropout = dropout
        self.only_gbdt = only_gbdt
        self.train_residual = train_non_gbdt
        self.name = name
        self.use_leaderboard = use_leaderboard
        self.iter_per_epoch = iter_per_epoch
        self.depth = depth
        self.lang = 'dgl'
        self.gbdt_lr = gbdt_lr
        self.gnn_dropout = 0.5
        self.gbdt_alpha = gbdt_alpha
        self.residual_before = None
        self.leaf_idx_before = [None, None, None]
        self.base_gbdt = [None, None, None]

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __name__(self):
        return 'ABGNN'

    def init_gbdt_model(self, num_epochs, epoch, m):
        if self.task[m] == 'regression':
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = 'RMSE' #''RMSEWithUncertainty'
        else:
            if epoch == 0:
                catboost_model_obj = CatBoostClassifier
                catboost_loss_fn = 'MultiClass'
            else:
                catboost_model_obj = CatBoostRegressor
                catboost_loss_fn = 'MultiRMSE'

        return catboost_model_obj(iterations=num_epochs,
                                  depth=self.depth,
                                  learning_rate=self.gbdt_lr,
                                  loss_function=catboost_loss_fn,
                                  random_seed=0,
                                  nan_mode='Min',
                                  allow_const_label=True)

    def fit_gbdt(self, pool, trees_per_epoch, epoch, m):
        gbdt_model = self.init_gbdt_model(trees_per_epoch, epoch, m)
        gbdt_model.fit(pool, verbose=False)
        return gbdt_model

    def init_gnn_model(self):
        self.model = [GNNModelDGL(in_dim=self.in_dim[m],
                                     hidden_dim=self.hidden_dim,
                                     out_dim=self.out_dim[m],
                                     name=self.name,
                                     dropout=self.dropout).to(self.device) for m in range(3)]

    def append_gbdt_model(self, new_gbdt_model, weights, m):
        if self.gbdt_model[m] is None:
            return new_gbdt_model
        return sum_models([self.gbdt_model[m], new_gbdt_model], weights=weights)

    def train_gbdt(self, gbdt_X_train, gbdt_y_train, cat_features, epoch,
                   gbdt_trees_per_epoch, gbdt_alpha, m):
        # print(gbdt_y_train.shape)
        pool = Pool(gbdt_X_train, gbdt_y_train, cat_features=cat_features)
        epoch_gbdt_model = self.fit_gbdt(pool, gbdt_trees_per_epoch, epoch, m)
        if epoch == 0 and self.task[m] == 'classification':
            self.base_gbdt[m] = epoch_gbdt_model
        else:
            self.gbdt_model[m] = self.append_gbdt_model(epoch_gbdt_model, weights=[1, gbdt_alpha], m=m)
            self.cur_gbdt = epoch_gbdt_model
    
    # predictions + leaf_index
    def update_node_features1(self, node_features, X, encoded_X, m):
        if self.task[m] == 'regression':
            predictions = np.expand_dims(self.gbdt_model[m].predict(X), axis=1)
            # predictions = self.gbdt_model[m]
        else:
            predictions = self.base_gbdt[m].predict_proba(X)
            if self.gbdt_model[m] is not None:
                predictions_after_one = self.gbdt_model[m].predict(X)[:, :self.out_dim[m]]
                predictions += predictions_after_one

        # MinMaxScaler
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        if self.leaf_idx_before[m] is None:
            if self.task[m] == 'classification':
                leaf_idx = self.base_gbdt[m].calc_leaf_indexes(X)
            else: 
                leaf_idx = self.gbdt_model[m].calc_leaf_indexes(X)
            
            leaf_idx_normalized = min_max_scaler.fit_transform(leaf_idx)
            self.leaf_idx_before[m] = leaf_idx
        else:        
            leaf_idx_cur = self.cur_gbdt.calc_leaf_indexes(X)
            self.leaf_idx_before[m] = np.append(self.leaf_idx_before[m], leaf_idx_cur, axis=1)

            leaf_idx_reshaped = self.leaf_idx_before[m].reshape(len(X), -1, self.iter_per_epoch).view()         
            leaf_idx = np.sum(leaf_idx_reshaped, axis=1)
            leaf_idx_normalized = min_max_scaler.fit_transform(leaf_idx)
        
        # print("leaf index before", self.leaf_idx_before[m][0])
        if not self.only_gbdt:
            # print(leaf_idx_normalized.shape)
            node_features_tem = np.append(predictions, leaf_idx_normalized, axis=1)

        node_features_tem = node_features_tem.astype("float")
        node_features_tem = torch.from_numpy(node_features_tem).to(self.device)

        node_features[m].data = node_features_tem.float().data
    
    # X + leaf index
    def update_node_features2(self, node_features, X, encocde_X, m):
        # MinMaxScaler
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        if self.leaf_idx_before[m] is None:
            if self.task[m] == 'classification':
                leaf_idx = self.base_gbdt[m].calc_leaf_indexes(X)
            else:
                leaf_idx = self.gbdt_model[m].calc_leaf_indexes(X)
            # print("leaf index 3", leaf_idx[0])
            leaf_idx_normalized = min_max_scaler.fit_transform(leaf_idx)
            self.leaf_idx_before[m] = leaf_idx
        else:
            # epoch leaf indexes
            leaf_idx_cur = self.cur_gbdt.calc_leaf_indexes(X)
            self.leaf_idx_before[m] = np.append(self.leaf_idx_before[m], leaf_idx_cur, axis=1)
            # print("leaf index before 3", self.leaf_idx_before[m][0])

            leaf_idx_reshaped = self.leaf_idx_before[m].reshape(len(X), -1, self.iter_per_epoch).view()         
            leaf_idx = np.sum(leaf_idx_reshaped, axis=1)
            # print(leaf_idx[0])
            leaf_idx_normalized = min_max_scaler.fit_transform(leaf_idx)
                
        if not self.only_gbdt:
            if self.train_residual:
                node_features_tem = np.append(node_features[m].detach().cpu().data[:, :-self.iter_per_epoch], leaf_idx_normalized, axis=1)
            else:
                node_features_tem = np.append(encocde_X, leaf_idx_normalized, axis=1)
            # print(node_features_tem.shape)

        node_features_tem = node_features_tem.astype("float")
        node_features_tem = torch.from_numpy(node_features_tem).to(self.device)
        node_features[m].data = node_features_tem.float().data


    # X + predictions
    def update_node_features3(self, node_features, X, encoded_X, m):
        if self.task[m] == 'regression':
            predictions = np.expand_dims(self.gbdt_model[m].predict(X), axis=1)
            # predictions = self.gbdt_model.virtual_ensembles_predict(X,
            #                                                         virtual_ensembles_count=5,
            #                                                         prediction_type='TotalUncertainty')
        else:
            predictions = self.base_gbdt[m].predict_proba(X)
            # print(predictions[0])
            if self.gbdt_model[m] is not None:
                predictions_after_one = self.gbdt_model[m].predict(X)
                predictions += predictions_after_one

        if not self.only_gbdt:
            if self.train_residual:
                predictions = np.append(node_features[m].detach().cpu().data[:, :-self.out_dim], predictions,
                                        axis=1)  # append updated X to prediction
            else:
                predictions = np.append(encoded_X, predictions, axis=1)  # append X to prediction
                # predictions = torch.cat([encoded_X, predictions], dim=1)

        predictions = torch.from_numpy(predictions).to(self.device)

        node_features[m].data = predictions.float().data

    def update_gbdt_targets(self, node_features, node_features_before, train_mask, m):
        if m == 0:
            residual = (node_features - node_features_before).detach().cpu().numpy()[train_mask, :]
        elif m == 1:
            residual = (node_features - node_features_before).detach().cpu().numpy()[train_mask, -self.iter_per_epoch:]
        else:
            residual = (node_features - node_features_before).detach().cpu().numpy()[train_mask, -self.out_dim[m]:]
        
        if self.task[m] == 'regression':
            residual = np.sum(residual, axis=1)
            
        return residual

    def init_node_features(self, X, m):
        node_features = torch.empty(X.shape[0], self.in_dim[m], requires_grad=True, device=self.device)
        if (not self.only_gbdt) and m!=0:
            node_features.data[:, :X.shape[1]] = torch.from_numpy(X.to_numpy(copy=True))
        return node_features   

    def init_node_parameters(self, num_nodes):
        return torch.empty(num_nodes, self.iter_per_epoch, requires_grad=True, device=self.device)

    def init_optimizer2(self, node_features, optimize_node_features, learning_rate, m):

        params = [self.model[m].parameters()]
        if optimize_node_features:
            params.append([node_features[m]])
        optimizer = torch.optim.Adam(itertools.chain(*params), lr=learning_rate)
        return optimizer

    def update_node_features_two(self, node_parameters, X):
        if self.task[m] == 'regression':
            predictions = np.expand_dims(self.gbdt_model.predict(X), axis=1)
        else:
            predictions = self.base_gbdt.predict_proba(X)
            if self.gbdt_model is not None:
                predictions += self.gbdt_model.predict(X)

        predictions = torch.from_numpy(predictions).to(self.device)
        node_parameters.data = predictions.float().data

    def layer_train_eval(self, model_in, y_residual, train_mask, val_mask, test_mask,
                         optimizer, m):
        loss = None

        for _ in range(self.iter_per_epoch):
            loss = self.train_models(model_in, y_residual, train_mask, optimizer, m)
        
        self.model[m].eval()
        layer_logits = self.model[m](*model_in).squeeze()
        train_results = self.evaluate_model(layer_logits, y_residual, train_mask, m)
        val_results = self.evaluate_model(layer_logits, y_residual, val_mask, m)
        test_results = self.evaluate_model(layer_logits, y_residual, test_mask, m)

        return loss, np.array([train_results, val_results, test_results])
    
    def evaluate_model(self, logits, target_labels, mask, m):
        metrics = {}
        y = target_labels[mask]
        with torch.no_grad():
            pred = logits[mask].float()
            print(type(pred))
            if self.task[m] == 'classification':
                try:
                    metrics['loss'] = F.cross_entropy(pred, y.long())
                except:
                    metrics['loss'] = F.cross_entropy(pred, y)
                metrics['accuracy'] = torch.Tensor([(y == pred.max(1)[1]).sum().item()/y.shape[0]])
            else:
                # metrics['loss'] = torch.sqrt(F.mse_loss(pred, y).squeeze() + 1e-8)
                metrics['loss'] = torch.sqrt(F.mse_loss(pred.view(-1), y.view(-1)))
                metrics['rmsle'] = torch.sqrt(F.mse_loss(torch.log(pred + 1), torch.log(y + 1)).squeeze() + 1e-8)
                metrics['mae'] = F.l1_loss(pred, y)
                metrics['r2'] = torch.Tensor([r2_score(y.cpu().numpy(), pred.cpu().numpy())])
            

            return metrics

    def train_models(self, model_in, target_labels, train_mask, optimizer, m):
        y = target_labels[train_mask]

        self.model[m].train()
        logits = self.model[m](*model_in).squeeze()
        pred = logits[train_mask]

        if self.task[m] == 'classification':
            # Adding softmax layer
            pred = F.softmax(pred, dim=-1)
            # calculate cross entropy
            loss = F.cross_entropy(pred, y.long())
        else:  # regression
            loss = torch.sqrt(F.mse_loss(pred.view(-1), y.view(-1)))        

        optimizer[m].zero_grad()
        loss.backward(retain_graph=True)
        optimizer[m].step()
        return loss
    
    def evaluate_metrics(self, logits, y, train_mask, val_mask, test_mask, metrics, m=0):
        train_results = self.evaluate_model(logits, y, train_mask, m)
        val_results = self.evaluate_model(logits, y, val_mask, m)
        test_results = self.evaluate_model(logits, y, test_mask, m)
        for metric_name in train_results:
            metrics[metric_name].append((train_results[metric_name].detach().item(),
                               val_results[metric_name].detach().item(),
                               test_results[metric_name].detach().item()
                               ))
    
    def get_layer_logits(self, model_in, m):
        # logits of the layer
        layer_logits = self.model[m](*model_in).squeeze()
        print("original layer_logits", layer_logits)

        if self.task[m] == 'classification':
            layer_logits = F.softmax(layer_logits, dim=1)[:, 1]
        
        return layer_logits
        

    def logits(self, X, target_labels, graph, encoded_X, node_features, node_features_before, gbdt_X_train, gbdt_y_train, cat_features, epoch,
               train_mask, optimizer):
        # first layer
        self.train_gbdt(gbdt_X_train, gbdt_y_train[0], cat_features, epoch,
                                self.iter_per_epoch, self.gbdt_alpha, 0)
        self.update_node_features[0](node_features, X, encoded_X, 0)
        node_features_before[0] = node_features[0].clone()
        model_in = (graph[0], node_features[0])
        y = target_labels[train_mask]
        logits1 = self.model[0](*model_in).squeeze()
        y_residuals = y - logits1

        # second layer
        self.train_gbdt(gbdt_X_train, gbdt_y_train[1], cat_features, epoch,
                                self.iter_per_epoch, self.gbdt_alpha, 1)
        self.update_node_features[1](node_features, X, encoded_X, 1)
        node_features_before[1] = node_features[1].clone()
        model_in = (graph[1], node_features[1])
        logits2 = self.model[1](*model_in).squeeze()
        y_residuals = y_residuals - logits2

        # third layer
        self.train_gbdt(gbdt_X_train, gbdt_y_train[2], cat_features, epoch,
                                self.iter_per_epoch, self.gbdt_alpha, 2)
        self.update_node_features[2](node_features, X, encoded_X, 2)
        node_features_before[2] = node_features[2].clone()
        model_in = (graph[2], node_features[2])
        logits3 = self.model[2](*model_in).squeeze()

        # aggregate logits
        logits = logits1 + logits2 + logits3
        # training
        # for _ in range(self.iter_per_epoch):
        #     loss = self.train_models(y, logits, optimizer)

        loss = self.train_models(y, logits, optimizer)

        y = target_labels[train_mask]
        y_residual = y.clone()
        logits = None
        for i in range(3):
            self.train_gbdt(gbdt_X_train, y_residual, cat_features, epoch,
                                self.iter_per_epoch, self.gbdt_alpha, 0)





    def fit(self, graph, graph_pred, graph_leaf, X, y, train_mask, val_mask, test_mask, cat_features,
            num_epochs, patience, logging_epochs=1, loss_fn=None, metric_name='loss',
            normalize_features=True, replace_na=True):

        # initialize for early stopping and metrics
        if metric_name in ['r2', 'accuracy']:
            best_metric = [np.float64('-inf')] * 3  # for train/val/test
        else:
            best_metric = [np.float64('inf')] * 3  # for train/val/test
        best_val_epoch = 0
        epochs_since_last_best_metric = 0
        metrics = ddict(list)
        if cat_features is None:
            cat_features = []

        if self.task == 'regression':
            self.out_dim = [y.shape[1] for _ in range(3)]
            self.task = ['regression', 'regression', 'regression']
        elif self.task == 'classification':
            # for binary task
            self.out_dim = [2, 1, 1]
            self.task = ['classification', 'regression', 'regression']
            
            # self.out_dim = len(set(y.iloc[test_mask, 0]))
        # self.in_dim = X.shape[1] if not self.only_gbdt else 0
        # self.in_dim += 3 if uncertainty else 1
            
        self.in_dim1 = self.out_dim[0] + self.iter_per_epoch  # Prediction + leaf index
        self.in_dim2 = X.shape[1] + self.iter_per_epoch  # X + leaf index 
        self.in_dim3 = X.shape[1] + self.out_dim[2]  # X + prediction
        self.in_dim = [self.in_dim1, self.in_dim2, self.in_dim3]

        # gnn model list consisting of 3 models
        self.init_gnn_model()

        gbdt_X_train = X.iloc[train_mask]
        gbdt_y_train = y.iloc[train_mask]
        
        self.gbdt_model = [None, None, None]

        encoded_X = X.copy()
        if not self.only_gbdt:
            if len(cat_features):
                encoded_X = self.encode_cat_features(encoded_X, y, cat_features, train_mask, val_mask, test_mask)
            if normalize_features:
                encoded_X = self.normalize_features(encoded_X, train_mask, val_mask, test_mask)
            if replace_na:
                encoded_X = self.replace_na(encoded_X, train_mask)

        # node feature list
        node_features = [self.init_node_features(encoded_X, m).to(self.device) for m in range(3)]
        node_features_before = [node_features[0].clone() for _ in range(3)]
        
        optimizer = [self.init_optimizer2(node_features, optimize_node_features=True, learning_rate=self.learning_rate, m=m) for m in range(3)]

        y, = self.pandas_to_torch(y)
        y_residual = y.clone()

        self.graph1 = graph.to(self.device)
        self.graph2 = graph_pred.to(self.device)
        self.graph3 = graph_leaf.to(self.device)
        self.graph = [self.graph1, self.graph2, self.graph3]

        self.update_node_features = [self.update_node_features1, self.update_node_features2, self.update_node_features3]

        pbar = tqdm(range(num_epochs))
        
        for epoch in pbar:
            start2epoch = time.time()
            logits = None
            loss = None
            layer_metric = []
            
            for m in range(3):
                for iter in range(self.iter_per_epoch):
                    self.train_gbdt(gbdt_X_train, gbdt_y_train, cat_features, iter, 
                                    self.iter_per_epoch, self.gbdt_alpha, m)
                    self.update_node_features[m](node_features, X, encoded_X, m)
                    node_features_before = node_features[m].clone()
                    print("node_features:", node_features[m].shape)
                    model_in = (self.graph[m], node_features[m])

                    layer_loss, metric_result = self.layer_train_eval(model_in, y_residual, train_mask, val_mask, 
                                                                test_mask, optimizer, m)
                    gbdt_y_train = self.update_gbdt_targets(node_features[m], node_features_before, train_mask, m) 
                    print("gbdt_y_train", gbdt_y_train.shape)

                layer_logits = self.get_layer_logits(model_in, m)
                layer_metric.append(metric_result)  # record layer metrics
                print(layer_metric)
                print('y_residual', y_residual)
                print('logits', layer_logits)
                y_residual = y_residual - layer_logits  # compute residual for next layer
                print('next', y_residual)
                gbdt_y_train = y_residual[train_mask].clone().detach().cpu().numpy()

                logits = layer_logits if logits is None else logits + layer_logits
                loss = layer_loss if loss is None else loss + layer_logits

            print("final logits:", logits)
            self.evaluate_metrics(logits, y, train_mask, val_mask, test_mask, metrics)
            self.log_epoch(pbar, metrics, epoch, loss, time.time() - start2epoch, logging_epochs,
                        metric_name=metric_name)
            # check early stopping
            best_metric, best_val_epoch, epochs_since_last_best_metric = \
                self.update_early_stopping(metrics, epoch, best_metric, best_val_epoch, epochs_since_last_best_metric,
                                        metric_name, lower_better=(metric_name not in ['r2', 'accuracy']))
            if patience and epochs_since_last_best_metric > patience:
                break
            if np.isclose(gbdt_y_train[2].sum(), 0.):
                print('Nodes do not change anymore. Stopping...')
                break

            if loss_fn:
                self.save_metrics(metrics, loss_fn)

        print('Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}'.format(metric_name, best_val_epoch, *best_metric))
        return metrics

    def predict(self, graph, X, y, test_mask):
        node_features = torch.empty(X.shape[0], self.in_dim).to(self.device)
        self.update_node_features(node_features, X, X)
        return self.evaluate_model((graph, node_features), y, test_mask)