#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from pycaret.classification import *
from sklearn.pipeline import Pipeline
import torch.nn as nn
from torch import optim
from skorch import NeuralNetClassifier
from skorch.helper import DataFrameTransformer

dat = pd.read_csv('data/data_for_model.csv', dtype = {'code': str})
processed = dat.drop(['diffEquity', 'diffAsset', 'diffNI', 'diffPNI'], axis=1)
processed = processed.loc[~ np.isnan(dat['diffRevenue']), ]
processed = processed.assign(diffRevenue = [1 if a > 0 else 0 for a in processed['diffRevenue']])
train = processed.query('fin_year < 2019')
test = processed.query('fin_year == 2019')
new = processed.query('fin_year == 2020')

clf1 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True,
             numeric_features=['fin_year'], numeric_imputation='mean',
             high_cardinality_features=['code'], high_cardinality_method='clustering',
             normalize=True, normalize_method='robust',
             transformation=True, transformation_method='yeo-johnson',
             fix_imbalance=True)
num = get_config('X_train').shape[1]

class Net(nn.Module):
    def __init__(self, num_inputs=12, num_units_d1=200, num_units_d2=100):
        super(Net, self).__init__()
        self.dense0 = nn.Linear(num, num_units_d1)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units_d1, num_units_d2)
        self.output = nn.Linear(num_units_d2, 2)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X

net = NeuralNetClassifier(
    module=Net,
    max_epochs=100,
    lr=0.1,
    batch_size=32,
    train_split=None
)

nn_pipe = Pipeline(
    [
        ("transform", DataFrameTransformer()),
        ("net", net),
    ]
)


skorch_model = create_model(nn_pipe, cross_validation=False)
pred_unseen = predict_model(skorch_model)
pull().to_csv('doc/output/pred_unseen.csv')
pred_unseen.to_csv('doc/output/prediction.csv')

compare_models(sort='AUC', include=['lr', 'rf', 'lightgbm', skorch_model], cross_validation=False)
pull().to_csv('doc/output/comparison.csv')


custom_grid = {
	'net__max_epochs':[20, 30],
	'net__lr': [0.01, 0.05, 0.1],
	'net__module__num_units_d1': [50, 100, 150, 200],
	'net__module__num_units_d2': [50, 100, 150, 200],
	'net__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}
tuned_skorch_model = tune_model(skorch_model, custom_grid=custom_grid)
compare_models(sort='AUC', include=['lr', 'rf', 'lightgbm', skorch_model, tuned_skorch_model], cross_validation=False)
pull().to_csv('doc/output/comparison.csv')

plot_model(tuned_skorch_model, plot='auc', scale=5, save='doc/figure/')
