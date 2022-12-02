#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from pycaret.classification import *

dat = pd.read_csv('data/data_for_model.csv', dtype = {'code': str})

processed = dat.drop(['diffEquity', 'diffAsset', 'diffNI', 'diffPNI'], axis=1)
processed = processed.loc[~ np.isnan(dat['diffRevenue']), ]
processed = processed.assign(diffRevenue = [1 if a > 0 else 0 for a in processed['diffRevenue']])
train = processed.query('fin_year < 2019')
test = processed.query('fin_year == 2019')
new = processed.query('fin_year == 2020')

clf1 = setup(data=train, test_data=test, target='diffRevenue', numeric_features=['fin_year'], html=False)
#  compare_models(sort='AUC', include=['lr', 'rf', 'lightgbm'])
best = compare_models(sort='AUC', cross_validation=False)
#  rf = create_model('rf', cross_validation=False)

#  save_model(rf, 'model/my_model')
#  rf = load_model('model/my_model')
#  plot_model(lr, plot='auc', save='doc/figure/')

evaluate_model(best)
#  pred_unseen = predict_model(rf, data=new)

