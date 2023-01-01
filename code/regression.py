#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------> import
import sys
import numpy as np
import pandas as pd
from pycaret.regression import *
import matplotlib
#  matplotlib.rcParams['font.family'] = ['SimHei']

# -----------------------------------------------------> init
dat = pd.read_csv('data/data_for_model.csv', dtype = {'code': str})
dat = dat.query('(diffPNI > -2) & (diffPNI < 2)')
processed = dat.drop(['diffEquity', 'diffAsset', 'diffNI', 'diffRevenue'], axis=1)
processed = processed.loc[~ np.isnan(dat['diffPNI']), ]
train = processed.query('fin_year < 2019')
test = processed.query('fin_year == 2019')
new = processed.query('fin_year == 2020')


# -----------------------------------------------------> preprocess
clf1 = setup(data=train, test_data=test, target='diffPNI', html=False, silent=True, session_id=1,
             imputation_type='simple', ignore_features=['fin_year'],
             high_cardinality_features=['code'], high_cardinality_method='clustering',
             normalize=True, normalize_method='robust',
             transformation=True, transformation_method='quantile')


# -----------------------------------------------------> train
best = compare_models(cross_validation=False)
pull().to_csv('doc/output/regression/comparison.csv')
best = create_model('catboost', cross_validation=False, return_train_score=True)
pull().to_csv('doc/output/regression/train_score.csv')
catboost = create_model('catboost', cross_validation=False)
lightgbm = create_model('lightgbm', cross_validation=False)
rf = create_model('rf', cross_validation=False)


# -----------------------------------------------------> plot
for plot_type in ['residuals', 'error', 'cooks', 'vc', 'manifold', 'parameter']:
    plot_model(best, plot=plot_type, scale=5, save='doc/figure/regression/')
    print('finish ' + plot_type)
for plot_type in ['summary', 'correlation', 'reason', 'pdp']: # pfi
    interpret_model(best, plot=plot_type, save='doc/figure/regression/')
    print('finish ' + plot_type)
#  deep_check(best)


# -----------------------------------------------------> plot long time
#  interpret_model(best, plot='pfi', save='doc/figure/regression/')
#  dt = create_model('dt', cross_validation=False)
#  plot_model(dt, plot='tree', save='doc/figure/regression/')
#  plot_model(lightgbm, plot='learning', save='doc/figure/regression/')
#  plot_model(lightgbm, plot='rfe', save='doc/figure/regression/')


# -----------------------------------------------------> ensemble
print('------start ensemble---------')
Boosting_catboost = ensemble_model(catboost, method='Boosting')
Boosting_lightgbm = ensemble_model(lightgbm, method='Boosting')
Boosting_rf = ensemble_model(rf, method='Boosting')
Bagging_catboost = ensemble_model(catboost, method='Bagging')
Bagging_lightgbm = ensemble_model(lightgbm, method='Bagging')
Bagging_rf = ensemble_model(rf, method='Bagging')
compare_models(include=[catboost, lightgbm, rf, Boosting_catboost, Boosting_lightgbm, Boosting_rf, Bagging_catboost, Bagging_lightgbm, Bagging_rf], cross_validation=False)
result = pull()
result.to_csv('doc/output/regression/result.csv')
result = pd.read_csv('doc/output/regression/result.csv', index_col=0)
model_type = pd.DataFrame({'type': ['Origin'] + [' '] * 2 + ['Boosting'] + [' '] * 2 + ['Bagging'] + [' '] * 2})
model_type.join(result.sort_index()).to_csv('doc/output/regression/ensemble.csv', index=False)


# -----------------------------------------------------> blend & stack
blender = blend_models([catboost, lightgbm, rf])
blender2 = blend_models([catboost, lightgbm, rf], weights=[0.7,0.2,0.1])
stacker = stack_models([catboost, lightgbm, rf], restack=False)
stacker2 = stack_models([catboost, lightgbm, rf], restack=True)
compare_models(include=[catboost, lightgbm, rf, blender, blender2, stacker, stacker2],
                      cross_validation=False)
result = pull()
result.to_csv('doc/output/regression/result.csv')
result = pd.read_csv('doc/output/regression/result.csv', index_col=0)
model_type = pd.DataFrame({'type': ['Origin'] + [' '] * 2 + ['Blending'] + ['Blending_weighted'] + ['Stacking'] + ['Stacking_restack']})
model_type.join(result.sort_index()).to_csv('doc/output/regression/stack.csv', index=False)


# -----------------------------------------------------> tune
best = tune_model(best, early_stopping='asha')


# -----------------------------------------------------> prediction
best = finalize_model(best)
prediction = predict_model(best, data=new)
pull().to_csv('doc/output/regression/pred_unseen.csv')
prediction.to_csv('doc/output/regression/prediction.csv')
prediction = pd.read_csv('doc/output/regression/prediction.csv')
total = prediction[['industry', 'diffPNI']].groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': '所有上市企业'})
top5 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(5).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': '前5'})
bot5 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(5).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': '后5'})
top10 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(10).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': '前10'})
bot10 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(10).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': '后10'})
top20 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(20).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': '前20'})
bot20 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(20).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': '后20'})
top30 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(30).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': '前30'})
bot30 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(30).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': '后30'})
merged = top5
for i in [top10, top20, top30, total, bot30, bot20, bot10, bot5]:
    merged = pd.merge(merged, i, on='industry', how='left')
merged = merged.rename(columns={'industry': '一级行业'})
merged.to_csv('doc/output/regression/portfolio.csv')


# -----------------------------------------------------> dashboard
clf1 = setup(data=train, test_data=test, target='diffPNI', html=False, silent=True, session_id=1,
             ignore_features=['fin_year', 'isST', 'industry', 'code', 'NRTurnRatio', 'INVTurnRatio', 'dupontIntburden'], numeric_imputation='mean')
catboost = create_model('catboost', cross_validation=False)
interpret_model(catboost, save='doc/figure/regression/')
#  dashboard(catboost)
