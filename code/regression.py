#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from pycaret.regression import *

dat = pd.read_csv('data/data_for_model.csv', dtype = {'code': str})
dat.loc[dat['diffPNI'] > 2, 'diffPNI'] = 2
processed = dat.drop(['diffEquity', 'diffAsset', 'diffNI', 'diffRevenue'], axis=1)
processed = processed.loc[~ np.isnan(dat['diffPNI']), ]
train = processed.query('fin_year < 2019')
test = processed.query('fin_year == 2019')
new = processed.query('fin_year == 2020')

clf1 = setup(data=train, test_data=test, target='diffPNI', html=False, silent=True,
             numeric_features=['fin_year'], numeric_imputation='mean',
             high_cardinality_features=['code'], high_cardinality_method='clustering',
             normalize=True, normalize_method='robust',
             transformation=True, transformation_method='yeo-johnson')

#  best = compare_models(cross_validation=False)
#  pull().to_csv('doc/output/regression/comparison.csv')
#  best = create_model(best, cross_validation=False, return_train_score=True)
#  best = create_model(best, return_train_score=True)
#  pull().to_csv('doc/output/regression/train_score.csv')


#  best = tune_model(best)
#  best = finalize_model(best)

#  plot_model(best, plot='manifold', save='doc/figure/')
#  interpret_model(best, plot='pdp', save='doc/figure/')
#  for plot_type in ['residuals', 'error', 'cooks', 'vc', 'manifold', 'feature', 'parameter']:
    #  plot_model(best, plot=plot_type, scale=5, save='doc/figure/regression/')
    #  print('finish ' + plot_type)
#  for plot_type in ['summary', 'correlation', 'reason', 'pdp', 'msa']: # pfi
    #  interpret_model(best, plot=plot_type, save='doc/figure/regression/')
    #  print('finish ' + plot_type)
#  deep_check(best)

pred_unseen = predict_model(best, data=new)
pull().to_csv('doc/output/regression/pred_unseen.csv')
pred_unseen.to_csv('doc/output/regression/prediction.csv')





prediction = pd.read_csv('doc/output/regression/prediction.csv')

total = prediction[['industry', 'diffPNI']].groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': 'ALL'})

top2 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(2).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': 'TOP2'})

bot2 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(2).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': 'BOT2'})

top5 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(5).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': 'TOP5'})

bot5 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(5).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': 'BOT5'})

top10 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(10).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': 'TOP10'})

bot10 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(10).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': 'BOT10'})

top20 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = False) \
    .groupby('industry').head(20).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': 'TOP20'})

bot20 = prediction[['industry', 'diffPNI', 'Label']].sort_values('Label', ascending = True) \
    .groupby('industry').head(20).groupby('industry').agg({'diffPNI': 'mean'}).rename(columns={'diffPNI': 'BOT20'})

merged = top2

for i in [top5, top10, top20, total, bot20, bot10, bot5, bot2]:
    merged = pd.merge(merged, i, on='industry', how='left')

merged.to_csv('doc/output/regression/portfolio.csv')

