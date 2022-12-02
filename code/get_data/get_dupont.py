#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import baostock as bs
import pandas as pd 

code_name = pd.read_csv("data/code_name.csv", dtype = str)
lg = bs.login()
for stock_code in code_name['code']:
    file_name = 'data/dupont/' + str(stock_code) + '.csv'
    if os.path.isfile(file_name):
        continue
    data_list = []
    for year in range(2000, 2022):
        rs = bs.query_dupont_data(code=stock_code, year=year, quarter=4)
        if rs.error_code != '0':
            print(rs.error_code)
            sys.exit()
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result.to_csv(file_name, index=False)
    print('get', stock_code)
#  bs.logout()
print('finish')

