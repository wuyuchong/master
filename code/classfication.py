#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from pycaret.classification import *

dat = pd.read_csv('data/data_for_model.csv', dtype = {'code': str})

processed = dat.drop(['diffEquity', 'diffAsset', 'diffNI', 'diffPNI'], axis=1)
processed = processed.loc[~ np.isnan(dat['diffRevenue']), ]
processed = processed.assign(diffRevenue = ['主营业务收入增长' if a > 0 else '主营业务收入减少' for a in processed['diffRevenue']])
train = processed.query('fin_year < 2019')
test = processed.query('fin_year == 2019')
new = processed.query('fin_year == 2020')

clf1 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True,
             numeric_features=['fin_year'], numeric_imputation='mean',
             high_cardinality_features=['code'], high_cardinality_method='clustering',
             normalize=True, normalize_method='robust',
             transformation=True, transformation_method='yeo-johnson',
             fix_imbalance=True)
best = compare_models(sort='AUC', cross_validation=False)
pull().to_csv('doc/output/classification/comparison.csv')
best = create_model(best, cross_validation=False)

#  save_model(rf, 'model/my_model')
#  rf = load_model('model/my_model')
for plot_type in ['auc', 'threshold', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'manifold', 'calibration', 'vc', 'dimension', 'feature', 'parameter', 'lift', 'gain', 'tree', 'ks']: # rfe learning
    plot_model(best, plot=plot_type, scale=5, save='doc/figure/classification/')
    print('finish ' + plot_type)

for plot_type in ['summary', 'correlation', 'reason', 'pdp', 'msa', 'pfi']:
    interpret_model(best, plot=plot_type, save='doc/figure/classification/')
    print('finish ' + plot_type)
#  plot_model(best, plot='auc', use_train_data = True, save='doc/figure/')

best = finalize_model(best)
best = tune_model(best)
#  evaluate_model(best)
pred_unseen = predict_model(best, data=new)
pull().to_csv('doc/output/classification/pred_unseen.csv')
pred_unseen.to_csv('doc/output/classification/prediction.csv')



prediction = pd.read_csv('doc/output/classification/prediction.csv')[['code', 'fin_year', 'Label']]
prediction = pd.merge(dat, prediction, on=['code', 'fin_year'])

total = prediction[['industry', 'diffRevenue']].groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'ALL'})

up = prediction[['industry', 'diffRevenue', 'Label']].query('Label == 1') \
    .groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'up'})

down = prediction[['industry', 'diffRevenue', 'Label']].query('Label == 0') \
    .groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'down'})

merged = pd.merge(up, total, on='industry', how='left')
merged = pd.merge(merged, down, on='industry', how='left')

merged.to_csv('doc/output/classification/portfolio.csv')
