import math

import pandas as pd
import numpy as np

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    # test:(User_id Merchant_id	Coupon_id	Discount_rate	Distance	Date_received)
    # df_test = pd.read_csv('../data/ccf_offline_stage1_test_revised.csv')

    # off线下消费数据:(User_id Merchant_id	Coupon_id	Discount_rate	Distance	Date_received	Date)
    df_off = pd.read_csv('../data/ccf_offline_stage1_train.csv', parse_dates=['Date_received', 'Date'])

    # print(df_off.head())
    # print(df_off.Merchant_id)

    X = df_off[:1000]
    cpu_jobs = 3

    stop = len(X)
    step = int(math.ceil(stop / cpu_jobs))

    X_chunks = [X[i:i + step] for i in range(0, stop, step)]
    X_list = [X] * cpu_jobs

    print(X_chunks)

    # on线上消费数据:(User_id	 Merchant_id	Action	Coupon_id	Discount_rate	Date_received	Date)
    # df_on = pd.read_csv('../data/ccf_online_stage1_train.csv')

    # 所有Coupon_id 为fixed 都被核销了，所以设置Discount_rate 为 0
    # print(df_on.loc[(df_on['Coupon_id'] == 'fixed')])
    # print(len(df_on.loc[(df_on['Coupon_id'] == 'fixed') & (df_on['Date'].notna())]))
    # print(len(df_on.loc[(df_on['Coupon_id'] == 'fixed') & (df_on['Date'].isna())]))

    # couponbydate = df_off[df_off['Date_received'].notnull()][['Date_received', 'Date']] \
    #     .groupby(['Date_received'], as_index=False).count()
    # couponbydate.columns = ['Date_received', 'count']
    # print(couponbydate)


