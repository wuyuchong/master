#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from pycaret.regression import *

dat = pd.read_csv('data/data_for_model.csv', dtype = {'code': str})

processed = dat.drop(['diffEquity', 'diffAsset', 'diffNI', 'diffPNI'], axis=1)
processed = processed.loc[~ np.isnan(dat['diffRevenue']), ]

processed.loc[processed['diffRevenue'] > 2, 'diffRevenue'] = 2

train = processed.query('fin_year < 2019')
test = processed.query('fin_year == 2019')
new = processed.query('fin_year == 2020')


clf1 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True,
             numeric_features=['fin_year'], numeric_imputation='mean',
             high_cardinality_features=['code'], high_cardinality_method='clustering',
             normalize=True, normalize_method='robust',
             transformation=True, transformation_method='yeo-johnson')
#  best = compare_models(cross_validation=False)
#  pull().to_csv('doc/output/comparison.csv')

xgboost = create_model('xgboost', cross_validation=False)
#  plot_model(xgboost, plot='manifold', save='doc/figure/')
#  interpret_model(xgboost, plot='pdp', save='doc/figure/')
for plot_type in ['residuals', 'error', 'cooks', 'vc', 'manifold', 'feature', 'parameter']:
    plot_model(xgboost, plot=plot_type, scale=5, save='doc/figure/regression/')
    print('finish ' + plot_type)

for plot_type in ['summary', 'correlation', 'reason', 'pdp', 'msa', 'pfi']:
    interpret_model(xgboost, plot=plot_type, save='doc/figure/regression/')
    print('finish ' + plot_type)
#  deep_check(xgboost)

pred_unseen = predict_model(xgboost, data=new)
pull().to_csv('doc/output/regression/pred_unseen.csv')
pred_unseen.to_csv('doc/output/regression/prediction.csv')





prediction = pd.read_csv('doc/output/prediction.csv')

total = prediction[['industry', 'diffRevenue']].groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'ALL'})

top2 = prediction[['industry', 'diffRevenue', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(2).groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'TOP2'})

bot2 = prediction[['industry', 'diffRevenue', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(2).groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'BOT2'})

top5 = prediction[['industry', 'diffRevenue', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(5).groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'TOP5'})

bot5 = prediction[['industry', 'diffRevenue', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(5).groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'BOT5'})

top10 = prediction[['industry', 'diffRevenue', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(10).groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'TOP10'})

bot10 = prediction[['industry', 'diffRevenue', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(10).groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'BOT10'})

top20 = prediction[['industry', 'diffRevenue', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(20).groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'TOP20'})

bot20 = prediction[['industry', 'diffRevenue', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(20).groupby('industry').agg({'diffRevenue': 'mean'}).rename(columns={'diffRevenue': 'BOT20'})

merged = top2

for i in [top5, top10, top20, total, bot20, bot10, bot5, bot2]:
    merged = pd.merge(merged, i, on='industry', how='left')

merged.to_csv('doc/output/portfolio.csv')
