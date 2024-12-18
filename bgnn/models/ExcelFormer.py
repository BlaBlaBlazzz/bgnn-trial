from __future__ import annotations

import torch
import time
import numpy as np
import pandas as pd
import torch_frame
import torch.nn.functional as F
from torch_frame.data import Dataset
from torch_frame.data.loader import DataLoader
from torch_frame.nn import ExcelFormer as Excel4ormer
from torch_frame.transforms import MutualInformationSort
from torch.optim.lr_scheduler import ExponentialLR
from collections import defaultdict as ddict

from .Base import BaseModel
from tqdm import tqdm

import copy
import logging
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from torch_frame import NAStrategy, TensorFrame, stype
from torch_frame.data.stats import StatType, compute_col_stats
from torch_frame.transforms import FittableBaseTransform

# updated cat to num transform feature
class CatToNumTransform(FittableBaseTransform):
    r"""Transforms categorical features in :class:`TensorFrame` using target
    statistics. The original transform is explained in
    `A preprocessing scheme for high-cardinality categorical attributes in
    classification and prediction problems
    <https://dl.acm.org/doi/10.1145/507533.507538>`_ paper.

    Specifically, each categorical feature is transformed into numerical
    feature using m-probability estimate, defined by

    .. math::
        \frac{n_c + p \cdot m}{n + m}

    where :math:`n_c` is the count of the category, :math:`n` is the total
    count, :math:`p` is the prior probability and :math:`m` is a smoothing
    factor.
    """
    def _fit(
        self,
        tf_train: TensorFrame,
        col_stats: dict[str, dict[StatType, Any]],
    ) -> None:
        if tf_train.y is None:
            raise RuntimeError(
                "'{self.__class__.__name__}' cannot be used when target column"
                " is None.")
        if stype.categorical not in tf_train.col_names_dict:
            logging.info(
                "The input TensorFrame does not contain any categorical "
                "columns. No fitting will be performed.")
            self._transformed_stats = col_stats
            return

        tensor = self._replace_nans(tf_train.feat_dict[stype.categorical],
                                    NAStrategy.MOST_FREQUENT)
        self.col_stats = col_stats
        columns = []
        self.data_size = tensor.size(0)
        # Check if it is multiclass classification task.
        # If it is multiclass classification task, then it doesn't make sense
        # to assume the target mean as the prior. Therefore, we need to expand
        # the number of columns to (num_target_classes - 1). More details can
        # be found in https://dl.acm.org/doi/10.1145/507533.507538
        if not torch.is_floating_point(tf_train.y) and tf_train.y.max() > 1:
            self.num_classes = tf_train.y.max() + 1
            target = F.one_hot(tf_train.y, self.num_classes)[:, :-1]
            self.target_mean = target.float().mean(dim=0)
            num_rows, num_cols = tf_train.feat_dict[stype.categorical].shape
            transformed_tensor = torch.zeros(num_rows,
                                             num_cols * (self.num_classes - 1),
                                             dtype=torch.float32,
                                             device=tf_train.device)
        else:
            self.num_classes = 2
            target = tf_train.y.unsqueeze(1)
            mask = ~torch.isnan(target)
            if (~mask).any():
                target = target[mask]
                if target.numel() == 0:
                    raise ValueError("Target value contains only nans.")
            self.target_mean = torch.mean(target.float())
            transformed_tensor = torch.zeros_like(
                tf_train.feat_dict[stype.categorical], dtype=torch.float32)

        for i in range(len(tf_train.col_names_dict[stype.categorical])):
            col_name = tf_train.col_names_dict[stype.categorical][i]
            count = torch.tensor(col_stats[col_name][StatType.COUNT][1],
                                 device=tf_train.device)
            feat = tensor[:, i]
            v = torch.index_select(count, 0, feat).unsqueeze(1).repeat(
                1, self.num_classes - 1)
            start = i * (self.num_classes - 1)
            end = (i + 1) * (self.num_classes - 1)
            transformed_tensor[:, start:end] = ((v + self.target_mean) /
                                                (self.data_size + 1))
            columns += [f"{col_name}_{i}" for i in range(self.num_classes - 1)]

        self.new_columns = columns
        transformed_df = pd.DataFrame(transformed_tensor.cpu().numpy(),
                                      columns=columns)

        transformed_col_stats = dict()
        if stype.numerical in tf_train.col_names_dict:
            for col in tf_train.col_names_dict[stype.numerical]:
                transformed_col_stats[col] = copy.copy(col_stats[col])
        for col in columns:
            # TODO: Make col stats computed purely with PyTorch
            # (without mapping back to pandas series).
            transformed_col_stats[col] = compute_col_stats(
                transformed_df[col], stype.numerical)

        self._transformed_stats = transformed_col_stats

    def _forward(self, tf: TensorFrame) -> TensorFrame:
        if stype.categorical not in tf.col_names_dict:
            logging.info(
                "The input TensorFrame does not contain any categorical "
                "columns. The original TensorFrame will be returned.")
            return tf
        tensor = self._replace_nans(
            tf.feat_dict[stype.categorical],
            NAStrategy.MOST_FREQUENT,
        )
        if not torch.is_floating_point(tf.y) and tf.y.max() > 1:
            num_rows, num_cols = tf.feat_dict[stype.categorical].shape
            transformed_tensor = torch.zeros(
                num_rows,
                num_cols * (self.num_classes - 1),
                dtype=torch.float32,
                device=tf.device,
            )
        else:
            transformed_tensor = torch.zeros_like(
                tf.feat_dict[stype.categorical],
                dtype=torch.float32,
            )
        target_mean = self.target_mean.to(tf.device)
        for i in range(len(tf.col_names_dict[stype.categorical])):
            col_name = tf.col_names_dict[stype.categorical][i]
            count = torch.tensor(
                self.col_stats[col_name][StatType.COUNT][1],
                device=tf.device,
            )
            feat = tensor[:, i]
            max_cat = feat.max()
            if max_cat >= len(count):
                raise RuntimeError(
                    f"'{col_name}' contains new category '{max_cat}' not seen "
                    f"during fit stage.")
            v = count[feat].unsqueeze(1).repeat(1, self.num_classes - 1)
            start = i * (self.num_classes - 1)
            end = (i + 1) * (self.num_classes - 1)
            transformed_tensor[:, start:end] = ((v + target_mean) /
                                                (self.data_size + 1))

        # turn the categorical features into numerical features
        if stype.numerical in tf.feat_dict:
            tf.feat_dict[stype.numerical] = torch.cat(
                (tf.feat_dict[stype.numerical], transformed_tensor),
                dim=1).to(torch.float32)
            tf.col_names_dict[stype.numerical] = tf.col_names_dict[
                stype.numerical] + self.new_columns
        else:
            tf.feat_dict[stype.numerical] = transformed_tensor
            tf.col_names_dict[stype.numerical] = self.new_columns
        # delete the categorical features
        tf.col_names_dict.pop(stype.categorical)
        tf.feat_dict.pop(stype.categorical)

        return tf




class ExcelFormer(BaseModel):
    def __init__(self, task='classification', 
                 in_channels=256, out_channels=1, num_heads=4, num_layers=5, lr=0.001,
                 gamma=0.95, beta=0.5, mixup=None, residual_dropout=0., diam_dropout=0.):
        super(ExcelFormer, self).__init__()
        self.task = task
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.learning_rate = lr
        self.gamma = gamma
        self.beta = beta
        self.mixup = mixup
        self.residual_dropout = residual_dropout
        self.diam_dropout = diam_dropout

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def __name__(self):
        return 'ExcelFormer'
    
    def train_val_test_split(self, data, train_mask, val_mask, test_mask):
        return data[train_mask], data[val_mask], data[test_mask]

    def data_loader(self, X, y, train_mask, val_mask, test_mask, cat_features):
        
        data = pd.concat([X, y], axis=1)
        col_name = data.columns.tolist()
        if cat_features is not None:
            cat_columns = [col_name[idx] for idx in cat_features]
        else:
            cat_columns = []
        
        if self.task == 'classification':
            cat_columns.append('class')
        
        col_to_stype = {col:torch_frame.numerical if col not in cat_columns else torch_frame.categorical for col in col_name}
        # col_to_stype = {col:torch_frame.numerical for col in col_name}

        data = Dataset(data, col_to_stype=col_to_stype, target_col="class")
        data.materialize()

        # split data
        train_data, val_data, test_data = self.train_val_test_split(data, train_mask, val_mask, test_mask)

        return train_data, val_data, test_data
    
    # batch size settings for datasets in (Grinsztajn et al., 2022)
    def get_batch_size(self, n_features):
        if n_features <= 32:
            batch_size = 512
            val_batch_size = 8192
        elif n_features <= 100:
            batch_size = 128
            val_batch_size = 512
        elif n_features <= 1000:
            batch_size = 32
            val_batch_size = 64
        else:
            batch_size = 16
            val_batch_size = 16
        
        return batch_size, val_batch_size

    def train_model(self, train_loader, optimizer):
        self.model.train()
        loss_sum = 0
        total_counts = 0

        for frame in train_loader:
            frame = frame.to(self.device)
            pred_mixed, y_mixed = self.model(frame, mixup_encoded=True)

            if self.task == "regression":
                loss = torch.sqrt(F.mse_loss(pred_mixed.view(-1), y_mixed.view(-1)))
            elif self.task == "classification":
                loss = F.cross_entropy(pred_mixed, y_mixed)
            else:
                raise NotImplemented("Unknown task. Supported tasks: classification, regression.")
            
            optimizer.zero_grad()
            loss.backward()
            loss_sum += float(loss) * len(y_mixed)
            total_counts += len(y_mixed)
            optimizer.step()
        
        return loss_sum / total_counts

    def model_evaluation(self, data_loader):
        metrics = {}
        total_counts = 0
        if self.task == "regression":
            metrics = {'loss':0, 'rmsle':0, 'mae':0, 'r2':0}
        elif self.task == 'classification':
            metrics = {'loss':0, 'accuracy':0}

        with torch.no_grad():
            # tensor frame of each batch
            for tf in data_loader:
                tf = tf.to(self.device)
                pred = self.model(tf)
                y = tf.y
                total_counts += len(y)
                if self.task == 'regression':
                    # print(F.mse_loss(pred, y))
                    metrics['loss'] += torch.sqrt(F.mse_loss(pred.view(-1), y.view(-1))) * len(y)
                    metrics['rmsle'] += torch.sqrt(F.mse_loss(torch.log(pred + 1), torch.log(y + 1)).squeeze() + 1e-8) * len(y)
                    metrics['mae'] += F.l1_loss(pred, y) * len(y)
                    metrics['r2'] += torch.Tensor([(y == pred.max(1)[1]).sum().item() / y.shape[0]]) * len(y)
                elif self.task == 'classification':
                    metrics['loss'] += F.cross_entropy(pred, y)
                    pred_class = pred.argmax(dim=-1)
                    metrics['accuracy'] += torch.Tensor([(y == pred_class).sum().item()])
            
            for key in metrics.keys():
                metrics[key] /= total_counts
            
            return metrics

    
    def train_and_evaluate(self, train_loader, val_loader, test_loader, optimizer, metrics):
        loss = None
        loss = self.train_model(train_loader, optimizer)
            
        self.model.eval()
        train_results = self.model_evaluation(train_loader)
        val_results = self.model_evaluation(val_loader)
        test_results = self.model_evaluation(test_loader)
        for metric_name in train_results:
            metrics[metric_name].append((train_results[metric_name].detach().item(),
                                        val_results[metric_name].detach().item(),
                                        test_results[metric_name].detach().item()))
        
        return loss


    def fit(self, X, y, train_mask, val_mask, test_mask, num_epochs,
            cat_features=None, patience=200, loss_fn="", metric_name="",
            logging_epochs=1, normalize_features=True, replace_na=True):
        
        # initialize for early stopping and metrics
        if metric_name in ['r2', 'accuracy']:
            best_metric = [np.float32('-inf')] * 3  # for train/val/test
        else:
            best_metric = [np.float32('inf')] * 3  # for train/val/test
        metrics = ddict(list)
        best_val_epoch = 0
        epochs_since_last_best_metric = 0

        X = X.copy()
        if cat_features is not None:
            encoded_X = self.encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask)
        # if normalize_features:
        #     X = self.normalize_features(X, train_mask, val_mask, test_mask)
        if replace_na:
            encoded_X = self.replace_na(X, train_mask)
        
        # load data
        train_data, val_data, test_data = self.data_loader(X, y, train_mask, val_mask, test_mask, cat_features)
        train_tensor_frame = train_data.tensor_frame
        val_tensor_frame = val_data.tensor_frame
        test_tensor_frame = test_data.tensor_frame

        categorical_transform = CatToNumTransform()
        categorical_transform.fit(train_tensor_frame, col_stats=train_data.col_stats)
        train_tensor_frame = categorical_transform(train_tensor_frame)
        val_tensor_frame = categorical_transform(val_tensor_frame)
        test_tensor_frame = categorical_transform(test_tensor_frame)
        col_stats = categorical_transform.transformed_stats
        
        mutual_info_sort = MutualInformationSort(task_type=train_data.task_type)
        mutual_info_sort.fit(train_tensor_frame, col_stats)
        train_tensor_frame = mutual_info_sort(train_tensor_frame)
        val_tensor_frame = mutual_info_sort(val_tensor_frame)
        test_tensor_frame = mutual_info_sort(test_tensor_frame)

        # batch size
        batch_size, val_batch_size = self.get_batch_size(len(train_data.feat_cols))

        # DataLoader
        train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_tensor_frame, batch_size=val_batch_size, shuffle=False)
        test_loader = DataLoader(test_tensor_frame, batch_size=val_batch_size, shuffle=False)
       
        # initialize ExcelFormer
        num_classes = max(y["class"].values.tolist()) + 1
        self.out_channels = int(num_classes) if self.task == 'classification' else 1
        self.model = Excel4ormer(in_channels=self.in_channels,
                            out_channels=self.out_channels,
                            num_cols=train_tensor_frame.num_cols,
                            num_layers=self.num_layers,
                            num_heads=self.num_heads,
                            mixup=self.mixup,
                            residual_dropout=self.residual_dropout,
                            diam_dropout=self.diam_dropout, 
                            col_stats=mutual_info_sort.transformed_stats,
                            col_names_dict=train_tensor_frame.col_names_dict).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=self.gamma)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start2epoch = time.time()

            loss = self.train_and_evaluate(train_loader, val_loader, test_loader,
                                            optimizer, metrics)
            self.log_epoch(pbar, metrics, epoch, loss, time.time()-start2epoch, logging_epochs,
                           metric_name=metric_name)
            
            best_metric, best_val_epoch, epochs_since_last_best_metric = \
                self.update_early_stopping(metrics, epoch, best_metric, best_val_epoch, epochs_since_last_best_metric, 
                                           metric_name, lower_better=(metric_name not in ['r2', 'accuracy']))
            if patience and epochs_since_last_best_metric > patience:
                break

            lr_scheduler.step()
        
        # if loss_fn:
        #     self.save_metrics(metrics, loss_fn)

        print('Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}'.format(metric_name, best_val_epoch, *best_metric))
        return metrics
            

        
        

        
