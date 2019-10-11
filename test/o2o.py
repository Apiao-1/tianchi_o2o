from datetime import date
import os, sys, pickle

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
# https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.6.29281b48FkMXnH&postId=8462


def getDiscountType(row):
    if pd.isnull(row):
        return np.nan
    elif ':' in row:
        return 1
    else:
        return 0


def convertRate(row):
    """Convert discount to rate"""
    if pd.isnull(row):
        return 1.0
    elif ':' in str(row):
        rows = row.split(':')
        return 1.0 - float(rows[1]) / float(rows[0])
    else:
        return 0.0  # rate为fixed时百分百核销


def getDiscountMan(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0


def getDiscountJian(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0


def processData(df):
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    # df['discount_rate'] = df.loc[:,'Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    # print(df['discount_rate'].unique())
    # convert distance
    try:
        df['distance'] = df['Distance'].fillna(-1).astype(int)
    except Exception as e:
        print(e)
    return df


def getWeekday(row):
    if row == 'nan':
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1


def label(row):
    if pd.isnull(row['Date_received']):
        return -1
    if pd.notnull(row['Date']):
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0


# 观察rate为fixed数据,发现均被核销
dt_fixed = []


def findFixed(row):
    # print(row)
    if "fixed" in str(row['Discount_rate']):
        # print(1)
        if row['Date_received'] != row['Date']:
            print(row['Name'])

def trainModel(dftrain, dfvalid, original_feature):
    print("----train-----")
    model = SGDClassifier(  # lambda:
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        max_iter=100,
        # shuffle=True,
        alpha=0.01,
        l1_ratio=0.01,
        n_jobs=1,
        class_weight=None
    )
    model.fit(dftrain[original_feature], dftrain['label'])

    # #### 预测以及结果评价
    print(model.score(dfvalid[original_feature], dfvalid['label']))

    print("---save model---")
    with open('1_model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    # test:(User_id Merchant_id	Coupon_id	Discount_rate	Distance	Date_received)
    df_test = pd.read_csv('../data/ccf_offline_stage1_test_revised.csv')
    # off线下消费数据:(User_id Merchant_id	Coupon_id	Discount_rate	Distance	Date_received	Date)
    df_off = pd.read_csv('../data/ccf_offline_stage1_train.csv')
    # on线上消费数据:(User_id	 Merchant_id	Action	Coupon_id	Discount_rate	Date_received	Date)
    df_on = pd.read_csv('../data/ccf_online_stage1_train.csv')
    # df_on.apply(findFixed, axis = 1) # axis = 0对所有列操作, 1对所有行
    # print(dt_fixed)
    # df_off = processData(df_off.head(10000))
    df_off = processData(df_off)
    # df_on = processData(df_on.head(10000))
    df_on = processData(df_on)
    df_test = processData(df_test)

    # 选出date_received中不为空的所有数据并排序
    # date_received = df_off['Date_received'].unique()
    # date_received = sorted(date_received[pd.notnull(date_received)])

    # date_buy = df_off['Date'].unique()
    # date_buy = sorted(date_buy[pd.notnull(date_buy)])
    date_buy = sorted(df_off[df_off['Date'].notnull()]['Date'])

    couponbydate = df_off[df_off['Date_received'].notnull()][['Date_received', 'Date']] \
        .groupby(['Date_received'], as_index=False).count()
    couponbydate.columns = ['Date_received', 'count']

    buybydate = df_off[(df_off['Date'].notnull()) & (df_off['Date_received'].notnull())][
        ['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
    buybydate.columns = ['Date_received', 'count']

    df_off['weekday'] = df_off['Date_received'].astype(str).apply(getWeekday)
    df_test['weekday'] = df_test['Date_received'].astype(str).apply(getWeekday)

    # weekday_type :  周六和周日为1，其他为0
    df_off['weekday_type'] = df_off['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)
    df_test['weekday_type'] = df_test['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)

    # change weekday to one-hot encoding
    weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
    tmpdf = pd.get_dummies(df_off['weekday'].replace('nan', np.nan))
    tmpdf.columns = weekdaycols
    df_off[weekdaycols] = tmpdf

    tmpdf = pd.get_dummies(df_test['weekday'].replace('nan', np.nan))
    tmpdf.columns = weekdaycols
    df_test[weekdaycols] = tmpdf

    df_off['label'] = df_off.apply(label, axis=1)  # 对所有行操作

    # data split
    print("-----data split------")
    df = df_off[df_off['label'] != -1].copy()
    train = df[(df['Date_received'] < 20160516)].copy()
    valid = df[(df['Date_received'] >= 20160516) & (df['Date_received'] <= 20160615)].copy()

    # feature
    original_feature = ['discount_rate', 'discount_type', 'discount_man', 'discount_jian', 'distance', 'weekday',
                        'weekday_type'] + weekdaycols

    trainModel(train, valid, original_feature)

    with open('1_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # test prediction for submission
    y_test_pred = model.predict_proba(df_test[original_feature])
    dftest1 = df_test[['User_id','Coupon_id','Date_received']].copy()
    dftest1['label'] = y_test_pred[:,1]
    dftest1.to_csv('submit1.csv', index=False, header=False)
    dftest1.head()