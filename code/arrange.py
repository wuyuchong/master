#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd

code_name = pd.read_csv('data/code_name.csv', dtype=str)
table = []

for stock_code in code_name['code']:
    file_name = 'data/profit/' + str(stock_code) + '.csv'
    financial = pd.read_csv(file_name)
    del file_name
    for category in ['operation', 'growth', 'balance', 'cash_flow', 'dupont']:
        file_name = 'data/' + category + '/' + str(stock_code) + '.csv'
        new = pd.read_csv(file_name)
        financial = pd.merge(financial, new, how='outer', on=['code', 'pubDate', 'statDate']) 

    financial['date'] = pd.to_datetime(financial['statDate'], format='%Y-%m-%d')
    financial['year'] = financial['date'].dt.year
    financial['fin_year'] = financial['year']
    financial['year1'] = financial['year'] + 1
    financial = financial.drop(['year', 'pubDate', 'statDate', 'date'], axis=1)

    filename = 'data/growth/' + str(stock_code) + '.csv'
    growth = pd.read_csv(filename)
    growth['date'] = pd.to_datetime(growth['statDate'], format='%Y-%m-%d')
    growth['year'] = growth['date'].dt.year
    growth = growth.drop(['YOYEPSBasic', 'code', 'pubDate', 'statDate', 'date'], axis=1)
    growth = growth.rename(columns={'YOYEquity': 'diffEquity', 'YOYAsset': 'diffAsset', 'YOYNI': 'diffNI', 'YOYPNI': 'diffPNI'})

    filename = 'data/profit/' + str(stock_code) + '.csv'
    profit = pd.read_csv(filename)
    profit['date'] = pd.to_datetime(profit['statDate'], format='%Y-%m-%d')
    profit['year'] = profit['date'].dt.year
    profit = profit[['year', 'MBRevenue']]
    profit = profit.rename(columns={'MBRevenue': 'MBRevenue2'})

    file_name = 'data/daily/' + str(stock_code) + '.csv'
    daily = pd.read_csv(file_name)
    del file_name
    daily['date'] = pd.to_datetime(daily['date'], format='%Y-%m-%d')
    daily = daily[daily['date'].dt.month == 5]
    daily['year'] = daily['date'].dt.year
    daily['day'] = daily['date'].dt.day
    daily['rank'] = daily.groupby('year')['day'].rank(method='first', ascending=True)
    daily = daily[daily['rank'] == 1]
    daily = daily[daily['tradestatus'] == 1]
    indicator = daily.drop(['date', 'code', 'close', 'day', 'rank', 'tradestatus'], axis=1)


    merge = pd.merge(financial, indicator, how='left', left_on='year1', right_on='year')
    del merge['year']
    merge = pd.merge(merge, profit, how='left', left_on='year1', right_on='year')
    del merge['year']
    merge = pd.merge(merge, growth, how='left', left_on='year1', right_on='year')
    del merge['year']
    del merge['year1']

    merge['diffRevenue'] = (merge['MBRevenue2'] - merge['MBRevenue']) / merge['MBRevenue']
    del merge['MBRevenue2']

    table.append(merge)

table = pd.concat(table, ignore_index=True)

industry = code_name[['code', 'industry']]
table = pd.merge(table, industry, how='left', on='code')

table.to_csv('data/data_for_model.csv', index=False)
print(table.head(3).transpose().to_string())
