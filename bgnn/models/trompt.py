import torch
import time
import numpy as np
import pandas as pd
import torch_frame
import torch.nn.functional as F
from torch_frame.data import Dataset
from torch_frame.data.loader import DataLoader
from torch_frame.nn import Trompt
from torch.optim.lr_scheduler import ExponentialLR
from collections import defaultdict as ddict

from .Base import BaseModel
from tqdm import tqdm




class trompt(BaseModel):
    def __init__(self, task='classification', 
                 channels=128, out_channels=1, num_layers=6, lr=0.001, gamma=0.95,
                 num_prompts=128):
        super(trompt, self).__init__()
        self.task = task
        self.channels = channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.learning_rate = lr
        self.gamma = gamma
        self.num_prompts = num_prompts

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def __name__(self):
        return 'ExcelFormer'
    
    def train_val_test_split(self, data, train_mask, val_mask, test_mask):
        return data[train_mask], data[val_mask], data[test_mask]

    def data_loader(self, X, y, train_mask, val_mask, test_mask):
        
        data = pd.concat([X, y], axis=1)
        col_name = data.columns.tolist()
        ctype = data.nunique().tolist() # verify categorical / numerical
        unique = {col:t for col, t in zip(col_name, ctype)}
        # print(unique)
        col_to_stype = {col:torch_frame.numerical if unique[col]>10 else torch_frame.categorical for col in col_name}
        # col_to_stype = {col:torch_frame.numerical for col in col_name}
        # print(col_to_stype)

        data = Dataset(data, col_to_stype=col_to_stype, target_col="class")
        data.materialize()

        # split data
        train_data, val_data, test_data = self.train_val_test_split(data, train_mask, val_mask, test_mask)

        return data, train_data, val_data, test_data
    
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
            out = self.model.forward_stacked(frame)
            num_layers = out.size(1)
            pred = out.view(-1, self.dataset.num_classes)
            y = frame.y.repeat_interleave(num_layers)

            if self.task == "regression":
                loss = torch.sqrt(F.mse_loss(pred, y))
            elif self.task == "classification":
                loss = F.cross_entropy(pred, y)
            else:
                raise NotImplemented("Unknown task. Supported tasks: classification, regression.")
            
            optimizer.zero_grad()
            loss.backward()
            loss_sum += float(loss) * len(frame.y)
            total_counts += len(frame.y)
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
                    metrics['loss'] += torch.sqrt(F.mse_loss(pred, y) + 1e8) * len(y)
                    metrics['rmsle'] += torch.sqrt(F.mse_loss(torch.log(pred + 1), torch.log(y + 1)).squeeze() + 1e-8) * len(y)
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


        if cat_features is not None:
            X = self.encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask)
        # if normalize_features:
        #     X = self.normalize_features(X, train_mask, val_mask, test_mask)
        if replace_na:
            X = X = self.replace_na(X, train_mask)
        
        # load data
        self.dataset, train_data, val_data, test_data = self.data_loader(X, y, train_mask, val_mask, test_mask)
        train_tensor_frame = train_data.tensor_frame
        val_tensor_frame = val_data.tensor_frame
        test_tensor_frame = test_data.tensor_frame

        # batch size
        batch_size, val_batch_size = self.get_batch_size(len(train_data.feat_cols))

        # DataLoader
        train_loader = DataLoader(train_tensor_frame, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_tensor_frame, batch_size=val_batch_size, shuffle=False)
        test_loader = DataLoader(test_tensor_frame, batch_size=val_batch_size, shuffle=False)
       
        # initialize Trompt
        self.model = Trompt(channels=self.channels,
                            out_channels=self.dataset.num_classes,
                            num_layers=self.num_layers,
                            num_prompts=self.num_prompts,
                            col_stats=self.dataset.col_stats,
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
        
        if loss_fn:
            self.save_metrics(metrics, loss_fn)

        print('Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}'.format(metric_name, best_val_epoch, *best_metric))
        return metrics
            

        
        

        
