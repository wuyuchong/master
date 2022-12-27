#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------> import
import sys
import numpy as np
import pandas as pd
from pycaret.classification import *
import matplotlib
matplotlib.rcParams['font.family'] = ['SimHei']


# -----------------------------------------------------> init
dat = pd.read_csv('data/data_for_model.csv', dtype = {'code': str})
dat.loc[dat['diffRevenue'] > 2, 'diffRevenue'] = 2
processed = dat.drop(['diffEquity', 'diffAsset', 'diffNI', 'diffPNI'], axis=1)
processed = processed.loc[~ np.isnan(dat['diffRevenue']), ]
processed = processed.assign(diffRevenue = ['up' if a > 0 else 'down' for a in processed['diffRevenue']])
train = processed.query('fin_year < 2019')
test = processed.query('fin_year == 2019')
new = processed.query('fin_year == 2020')


# -----------------------------------------------------> preprocess
#  clf0 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             #  numeric_features=['fin_year'], numeric_imputation='mean',
             #  ignore_features=['code', 'industry'])
#  compare_models(cross_validation=False, include=['lr', 'ridge', 'knn', 'svm'])
#  compare_models(cross_validation=False, include=['catboost', 'lightgbm', 'xgboost', 'rf'])
#  before = pull().sort_index()
#  clf0 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             #  numeric_features=['fin_year'], numeric_imputation='mean',
             #  high_cardinality_features=['code'], high_cardinality_method='clustering',
             #  ignore_features=['industry'])
#  compare_models(cross_validation=False, include=['lr', 'ridge', 'knn', 'svm'])
#  compare_models(cross_validation=False, include=['catboost', 'lightgbm', 'xgboost', 'rf'])
#  after = pull().sort_index()
#  result = pd.concat([before, after]).reset_index()
#  model_type = pd.DataFrame({'type': ['Origin'] * 4 + ['imputation'] * 4})
#  result = pd.concat([model_type, result], axis=1)
#  result.to_csv('doc/output/classification/preprocess.csv')



# -----------------------------------------------------> train
clf1 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             numeric_features=['fin_year'], imputation_type='simple',
             high_cardinality_features=['code'], high_cardinality_method='clustering',
             normalize=True, normalize_method='robust',
             transformation=True, transformation_method='yeo-johnson',
             fix_imbalance=True)
#  best = compare_models(sort='AUC', cross_validation=False)
#  pull().to_csv('doc/output/classification/comparison.csv')
#  best = create_model('catboost', cross_validation=False, return_train_score=True)
#  best = create_model(best, return_train_score=True)
#  pull().to_csv('doc/output/classification/train_score.csv')


# -----------------------------------------------------> plot
#  plot_model(best, plot='auc', use_train_data = True, save='doc/figure/')
#  for plot_type in ['auc', 'threshold', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'manifold', 'vc', 'dimension', 'feature', 'parameter', 'lift', 'gain', 'ks']: # rfe learning tree
    #  plot_model(best, plot=plot_type, scale=5, save='doc/figure/classification/')
    #  print('finish ' + plot_type)
#  for plot_type in ['summary', 'correlation', 'reason', 'pdp', 'msa']: # pfi
    #  interpret_model(best, plot=plot_type, save='doc/figure/classification/')
    #  print('finish ' + plot_type)


catboost = create_model('catboost', cross_validation=False)
lightgbm = create_model('lightgbm', cross_validation=False)
xgboost = create_model('xgboost', cross_validation=False)


# -----------------------------------------------------> ensemble
#  print('------start ensemble---------')
#  Boosting_catboost = ensemble_model(catboost, method='Boosting')
#  Boosting_lightgbm = ensemble_model(lightgbm, method='Boosting')
#  Boosting_xgboost = ensemble_model(xgboost, method='Boosting')
#  Bagging_catboost = ensemble_model(catboost, method='Bagging')
#  Bagging_lightgbm = ensemble_model(lightgbm, method='Bagging')
#  Bagging_xgboost = ensemble_model(xgboost, method='Bagging')
#  best = compare_models(include=[catboost, lightgbm, xgboost, Boosting_catboost, Boosting_lightgbm, Boosting_xgboost, Bagging_catboost, Bagging_lightgbm, Bagging_xgboost],
                      #  cross_validation=False)
#  result = pull()
#  result.to_csv('doc/output/classification/result.csv')
#  result = pd.read_csv('doc/output/classification/result.csv', index_col=0)
#  model_type = pd.DataFrame({'type': ['Origin'] * 3 + ['Boosting'] * 3 + ['Bagging'] * 3})
#  model_type.join(result.sort_index()).to_csv('doc/output/classification/ensemble.csv', index=False)
#  best = create_model(best, cross_validation=False, return_train_score=True)


# -----------------------------------------------------> blend & stack
blender = blend_models([catboost, lightgbm, xgboost])
stacker = stack_models([catboost, lightgbm, xgboost])
stacker2 = stack_models([catboost, lightgbm, xgboost], restack='False')
best = compare_models(include=[catboost, lightgbm, xgboost, blender, stacker],
                      cross_validation=False)
result = pull()
result.to_csv('doc/output/classification/result.csv')
result = pd.read_csv('doc/output/classification/result.csv', index_col=0)
model_type = pd.DataFrame({'type': ['Origin'] * 3 + ['Blending'] + ['Stacking']})
model_type.join(result.sort_index()).to_csv('doc/output/classification/stack.csv', index=False)


# -----------------------------------------------------> tune
#  best = tune_model(best, early_stopping='asha')
# 参数调整具体参见手册


# -----------------------------------------------------> calibration
#  plot_model(best, plot='calibration', save='doc/figure/classification/before')
#  best = calibrate_model(best)
#  plot_model(best, plot='calibration', save='doc/figure/classification')
#  best = finalize_model(best)


# -----------------------------------------------------> prediction
#  pred_unseen = predict_model(best, data=new)
#  pull().to_csv('doc/output/classification/pred_unseen.csv')
#  pred_unseen.to_csv('doc/output/classification/prediction.csv')
#  prediction = pd.read_csv('doc/output/classification/prediction.csv')[['code', 'fin_year', 'Label']]
#  prediction = pd.merge(dat, prediction, on=['code', 'fin_year'])
#  total = prediction[['industry', 'diffRevenue']].groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'ALL'})
#  up = prediction[['industry', 'diffRevenue', 'Label']].query('Label == "up"') \
    #  .groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'up'})
#  down = prediction[['industry', 'diffRevenue', 'Label']].query('Label == "down"') \
    #  .groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'down'})
#  merged = pd.merge(up, total, on='industry', how='left')
#  merged = pd.merge(merged, down, on='industry', how='left')
#  merged.to_csv('doc/output/classification/portfolio.csv')
