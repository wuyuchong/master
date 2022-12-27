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
clf0 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             numeric_features=['fin_year'], imputation_type='simple',
             ignore_features=['code', 'industry'])
compare_models(cross_validation=False, include=['lr', 'ridge', 'knn'])
before = pull().sort_index()


# -----------------------------------------------------> preprocess imputation
clf0 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             numeric_features=['fin_year'], imputation_type='iterative',
             ignore_features=['code', 'industry'])
compare_models(cross_validation=False, include=['lr', 'ridge', 'knn'])
after = pull().sort_index()
result = pd.concat([before, after]).reset_index().drop('Model', axis=1).rename(columns={'index': 'Model'})
model_type = pd.DataFrame({'Type': ['单变量插补法'] + [' '] * 2 + ['多变量插补法'] + [' '] * 2})
result = pd.concat([model_type, result], axis=1)
result.to_csv('doc/output/classification/preprocess_imputation.csv', index=False)


# -----------------------------------------------------> preprocess normalize
#  clf0 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             #  numeric_features=['fin_year'], imputation_type='simple',
             #  normalize=True, normalize_method='robust',
             #  ignore_features=['code', 'industry'])
#  compare_models(cross_validation=False, include=['lr', 'ridge', 'knn'])
#  after = pull().sort_index()
#  result = pd.concat([before, after]).reset_index().drop('Model', axis=1).rename(columns={'index': 'Model'})
#  model_type = pd.DataFrame({'Type': ['未处理'] + [' '] * 2 + ['分位数标准化'] + [' '] * 2})
#  result = pd.concat([model_type, result], axis=1)
#  result.to_csv('doc/output/classification/preprocess_normalize.csv', index=False)


# -----------------------------------------------------> preprocess transformation
clf0 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             numeric_features=['fin_year'], imputation_type='simple',
             normalize=True, normalize_method='robust',
             transformation=True, transformation_method='yeo-johnson',
             ignore_features=['code', 'industry'])
compare_models(cross_validation=False, include=['lr', 'ridge', 'knn'])
after = pull().sort_index()
clf0 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             numeric_features=['fin_year'], imputation_type='simple',
             normalize=True, normalize_method='robust',
             transformation=True, transformation_method='quantile',
             ignore_features=['code', 'industry'])
compare_models(cross_validation=False, include=['lr', 'ridge', 'knn'])
after2 = pull().sort_index()
result = pd.concat([before, after, after2]).reset_index().drop('Model', axis=1).rename(columns={'index': 'Model'})
model_type = pd.DataFrame({'Type': ['未处理'] + [' '] * 2 + ['yeo-johnson'] + [' '] * 2 + ['quantile'] + [' '] * 2})
result = pd.concat([model_type, result], axis=1)
result.to_csv('doc/output/classification/preprocess_transformation.csv', index=False)


# -----------------------------------------------------> preprocess imbalance
clf0 = setup(data=train, test_data=test, target='diffRevenue', html=False, silent=True, session_id=1,
             numeric_features=['fin_year'], imputation_type='iterative',
             ignore_features=['code', 'industry'],
             fix_imbalance=True)
compare_models(cross_validation=False, include=['lr', 'ridge', 'knn'])
after = pull().sort_index()
result = pd.concat([before, after]).reset_index().drop('Model', axis=1).rename(columns={'index': 'Model'})
model_type = pd.DataFrame({'Type': ['未处理'] + [' '] * 2 + ['SMOTE过采样'] + [' '] * 2})
result = pd.concat([model_type, result], axis=1)
result.to_csv('doc/output/classification/preprocess_imbalance.csv', index=False)
