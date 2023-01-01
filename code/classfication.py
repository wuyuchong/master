#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------> import
import sys
import numpy as np
import pandas as pd
from pycaret.classification import *
import matplotlib
#  matplotlib.rcParams['font.family'] = ['SimHei']


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
clf1 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             imputation_type='simple', ignore_features=['fin_year'],
             high_cardinality_features=['code'], high_cardinality_method='clustering',
             normalize=True, normalize_method='robust',
             transformation=True, transformation_method='quantile',
             fix_imbalance=True)


# -----------------------------------------------------> train
compare_models(sort='AUC', cross_validation=False)
pull().to_csv('doc/output/classification/comparison.csv')
best = create_model('catboost', cross_validation=False, return_train_score=True)
pull().to_csv('doc/output/classification/train_score.csv')
catboost = create_model('catboost', cross_validation=False)
lightgbm = create_model('lightgbm', cross_validation=False)
xgboost = create_model('xgboost', cross_validation=False)


# -----------------------------------------------------> plot
for plot_type in ['auc', 'threshold', 'pr', 'confusion_matrix', 'error', 'class_report', 'manifold', 'vc', 'dimension', 'parameter', 'lift', 'gain', 'ks']: # rfe learning tree
    plot_model(best, plot=plot_type, scale=5, save='doc/figure/classification/')
    print('finish ' + plot_type)
for plot_type in ['summary', 'correlation', 'reason', 'pdp']: # pfi
    interpret_model(best, plot=plot_type, save='doc/figure/classification/')
    print('finish ' + plot_type)
#  deep_check(best)
#  use_train_data


# -----------------------------------------------------> ensemble
#  print('------start ensemble---------')
#  Boosting_catboost = ensemble_model(catboost, method='Boosting')
#  Boosting_lightgbm = ensemble_model(lightgbm, method='Boosting')
#  Boosting_xgboost = ensemble_model(xgboost, method='Boosting')
#  Bagging_catboost = ensemble_model(catboost, method='Bagging')
#  Bagging_lightgbm = ensemble_model(lightgbm, method='Bagging')
#  Bagging_xgboost = ensemble_model(xgboost, method='Bagging')
#  compare_models(include=[catboost, lightgbm, xgboost, Boosting_catboost, Boosting_lightgbm, Boosting_xgboost, Bagging_catboost, Bagging_lightgbm, Bagging_xgboost], cross_validation=False)
#  result = pull()
#  result.to_csv('doc/output/classification/result.csv')
#  result = pd.read_csv('doc/output/classification/result.csv', index_col=0)
#  model_type = pd.DataFrame({'type': ['Origin'] + [' '] * 2 + ['Boosting'] + [' '] * 2 + ['Bagging'] + [' '] * 2})
#  model_type.join(result.sort_index()).to_csv('doc/output/classification/ensemble.csv', index=False)


# -----------------------------------------------------> blend & stack
#  blender = blend_models([catboost, lightgbm, xgboost], method='soft')
#  blender2 = blend_models([catboost, lightgbm, xgboost], method='hard')
#  stacker = stack_models([catboost, lightgbm, xgboost], restack=False)
#  stacker2 = stack_models([catboost, lightgbm, xgboost], restack=True)
#  compare_models(include=[catboost, lightgbm, xgboost, blender, blender2, stacker, stacker2],
                      #  cross_validation=False)
#  result = pull()
#  result.to_csv('doc/output/classification/result.csv')
#  result = pd.read_csv('doc/output/classification/result.csv', index_col=0)
#  model_type = pd.DataFrame({'type': ['Origin'] + [' '] * 2 + ['Blending_soft'] + ['Blending_hard'] + ['Stacking'] + ['Stacking_restack']})
#  model_type.join(result.sort_index()).to_csv('doc/output/classification/stack.csv', index=False)


# -----------------------------------------------------> tune
best = tune_model(best, early_stopping='asha')
# 参数调整具体参见手册


# -----------------------------------------------------> calibration
#  rf = create_model('rf', cross_validation=False)
#  plot_model(rf, plot='calibration', save='doc/figure/classification/before')
#  rf = calibrate_model(rf)
#  plot_model(rf, plot='calibration', save='doc/figure/classification')


# -----------------------------------------------------> prediction
best = finalize_model(best)
pred_unseen = predict_model(best, data=new)
pull().to_csv('doc/output/classification/pred_unseen.csv')
pred_unseen.to_csv('doc/output/classification/prediction.csv')
prediction = pd.read_csv('doc/output/classification/prediction.csv')[['code', 'fin_year', 'Label']]
prediction = pd.merge(dat, prediction, on=['code', 'fin_year'])
total = prediction[['industry', 'diffRevenue']].groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': '所有上市企业'})
up = prediction[['industry', 'diffRevenue', 'Label']].query('Label == "up"') \
    .groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': '预测增长'})
down = prediction[['industry', 'diffRevenue', 'Label']].query('Label == "down"') \
    .groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': '预测下降'})
merged = pd.merge(up, total, on='industry', how='left')
merged = pd.merge(merged, down, on='industry', how='left')
merged = merged.rename(columns={'industry': '一级行业'})
merged.to_csv('doc/output/classification/portfolio.csv')


# -----------------------------------------------------> plot special
#  clf0 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             #  imputation_type='simple', fix_imbalance=True,
             #  ignore_features=['code', 'industry'])
#  lr = create_model('lr', cross_validation=False, return_train_score=True)
#  plot_model(lr, plot='boundary', scale=5, save='doc/figure/classification/')

# 变量重要性图 code industry 不要了


# -----------------------------------------------------> dashboard
clf1 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             ignore_features=['fin_year', 'isST', 'industry', 'code', 'NRTurnRatio', 'INVTurnRatio', 'dupontIntburden'],
             numeric_imputation='mean', fix_imbalance=True)
catboost = create_model('catboost', cross_validation=False)
interpret_model(catboost, save='doc/figure/classification/')
#  dashboard(catboost)





