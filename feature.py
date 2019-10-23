import datetime
import os
import time
from concurrent.futures import ProcessPoolExecutor
from math import ceil

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from xgboost.sklearn import XGBClassifier
from tianchi_o2o import model
from tianchi_o2o import logger
import seaborn as sns

date_null = pd.to_datetime('1970-01-01', format='%Y-%m-%d')
cpu_jobs = cpu_jobs = os.cpu_count() - 1

def drop_columns(X, predict=False):
    columns = [
        'User_id', 'Merchant_id', 'Discount_rate', 'Date_received', 'discount_rate_x', 'discount_rate_y',
        # 'u33', 'u34'
    ]

    if predict:
        columns.append('Coupon_id')
    else:
        columns.append('Date')

    X.drop(columns=columns, inplace=True)


def get_preprocess_data(predict=False):
    if predict:
        offline = pd.read_csv('data/ccf_offline_stage1_test_revised.csv', parse_dates=['Date_received'])
    else:
        offline = pd.read_csv('data/ccf_offline_stage1_train.csv', parse_dates=['Date_received', 'Date'])

    offline.Distance.fillna(11, inplace=True)  # 用11填充缺失值
    offline.Distance = offline.Distance.astype(int)  # 转换数据类型为int
    offline.Coupon_id.fillna(0, inplace=True)  # 填充
    offline.Coupon_id = offline.Coupon_id.astype(int)  # 转换数据类型
    global date_null
    offline.Date_received.fillna(date_null, inplace=True)  # 填充缺失值

    offline[['discount_rate_x', 'discount_rate_y']] = offline[offline.Discount_rate.str.contains(':') == True][
        'Discount_rate'].str.split(':', expand=True).astype(int)
    offline['discount_rate'] = 1 - offline.discount_rate_y / offline.discount_rate_x

    offline.discount_rate = offline.discount_rate.fillna(offline.Discount_rate).astype(float)

    if predict:
        return offline

    offline.Date.fillna(date_null, inplace=True)

    # online
    online = pd.read_csv('data/ccf_online_stage1_train.csv', parse_dates=['Date_received', 'Date'])

    online.Coupon_id.fillna(0, inplace=True)
    # online.Coupon_id = online.Coupon_id.astype(int)
    online.Date_received.fillna(date_null, inplace=True)
    online.Date.fillna(date_null, inplace=True)

    return offline, online


def get_train_data():
    global date_null
    path = 'cache_%s_train.csv' % os.path.basename(__file__)

    if os.path.exists(path):
        data = pd.read_csv(path)
        # print(len(data))
    else:
        offline, online = get_preprocess_data()

        # date received 2016-01-01 - 2016-06-15
        # date consumed 2016-01-01 - 2016-06-30

        # train data 1
        # 2016-04-16 ~ 2016-05-15
        data_1 = offline[('2016-04-16' <= offline.Date_received) & (offline.Date_received <= '2016-05-15')].copy()
        # 这里是二分类，15天内消耗了优惠券就是1，未消耗则是0，删去了所有未领券的数据
        data_1['label'] = 0
        data_1.loc[
            (data_1.Date != date_null) & (data_1.Date - data_1.Date_received <= datetime.timedelta(15)), 'label'] = 1

        # feature data 1
        # 领券 2016-01-01 ~ 2016-03-31
        end = '2016-03-31'
        data_off_1 = offline[offline.Date_received <= end]
        data_on_1 = online[online.Date_received <= end]

        # 普通消费 2016-01-01 ~ 2016-04-15
        end = '2016-04-15'
        data_off_2 = offline[(offline.Coupon_id == 0) & (offline.Date <= end)]
        data_on_2 = online[(online.Coupon_id == 0) & (online.Date <= end)]

        data_1 = get_offline_features(data_1, pd.concat([data_off_1, data_off_2]))
        data_1 = get_online_features(pd.concat([data_on_1, data_on_2]), data_1)

        # train data 2
        # 2016-05-16 ~ 2016-06-15
        data_2 = offline['2016-05-16' <= offline.Date_received].copy()
        data_2['label'] = 0
        data_2.loc[
            (data_2.Date != date_null) & (data_2.Date - data_2.Date_received <= datetime.timedelta(15)), 'label'] = 1

        # feature data 2
        # 领券
        start = '2016-02-01'
        end = '2016-04-30'
        data_off_1 = offline[(start <= offline.Date_received) & (offline.Date_received <= end)]
        data_on_1 = online[(start <= online.Date_received) & (online.Date_received <= end)]

        # 普通消费
        start = '2016-02-01'
        end = '2016-05-15'
        data_off_2 = offline[(offline.Coupon_id == 0) & (start <= offline.Date) & (offline.Date <= end)]
        data_on_2 = online[(online.Coupon_id == 0) & (start <= online.Date) & (online.Date <= end)]

        data_2 = get_offline_features(data_2, pd.concat([data_off_1, data_off_2]))
        data_2 = get_online_features(pd.concat([data_on_1, data_on_2]), data_2)

        data = pd.concat([data_1, data_2])  # 数据合并与重塑

        # undersampling
        # if undersampling:
        #     temp = X_1[X_1.label == 1].groupby('User_id').size().reset_index()
        #     temp = X_1[X_1.User_id.isin(temp.User_id)]
        #     X_1 = pd.concat([temp, X_1[~X_1.User_id.isin(temp.User_id)].sample(4041)])

        # data.drop_duplicates(inplace=True)
        drop_columns(data)
        data.fillna(0, inplace=True)
        data.to_csv(path, index=False)

    return data

# 分成两部分的原因是，这里的X不会涉及其Date属性，保持其和待预测集一样
def get_offline_features(X, offline):
    # X = X[:1000]
    global date_null

    print(len(X), len(X.columns))
    print("X.head(5)", X.head(5))
    print("offline.head(5)", offline.head(5))

    # 数据量非常大，提取特征很慢时，可以参考分表的思想，避免重复计算
    temp = offline[offline.Coupon_id != 0] # 领了券的用户
    coupon_consume = temp[temp.Date != date_null] # 领了券消费的
    coupon_no_consume = temp[temp.Date == date_null] # 领了券没消费的

    user_coupon_consume = coupon_consume.groupby('User_id')

    # 一周中的第几天
    X['weekday'] = X.Date_received.dt.weekday
    # 一个月中的第几天
    X['day'] = X.Date_received.dt.day


    # # 距离优惠券消费次数
    # temp = coupon_consume.groupby('Distance').size().reset_index(name='distance_0')
    # X = pd.merge(X, temp, how='left', on='Distance')
    #
    # # 距离优惠券不消费次数
    # temp = coupon_no_consume.groupby('Distance').size().reset_index(name='distance_1')
    # X = pd.merge(X, temp, how='left', on='Distance')
    #
    # # 距离优惠券领取次数
    # X['distance_2'] = X.distance_0 + X.distance_1
    #
    # # 距离优惠券消费率
    # X['distance_3'] = X.distance_0 / X.distance_2

    # temp = coupon_consume[coupon_consume.Distance != 11].groupby('Distance').size()
    # temp['d4'] = temp.Distance.sum() / len(temp)
    # X = pd.merge(X, temp, how='left', on='Distance')

    '''user features'''

    # 优惠券消费次数
    temp = user_coupon_consume.size().reset_index(name='u2')
    X = pd.merge(X, temp, how='left', on='User_id')
    # X.u2.fillna(0, inplace=True)
    # X.u2 = X.u2.astype(int)

    # 优惠券不消费次数
    temp = coupon_no_consume.groupby('User_id').size().reset_index(name='u3')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 使用优惠券次数与没使用优惠券次数比值
    X['u19'] = X.u2 / X.u3

    # 领取优惠券次数
    X['u1'] = X.u2.fillna(0) + X.u3.fillna(0)

    # 优惠券核销率
    X['u4'] = X.u2 / X.u1

    # 普通消费次数
    temp = offline[(offline.Coupon_id == 0) & (offline.Date != date_null)]
    temp1 = temp.groupby('User_id').size().reset_index(name='u5')
    X = pd.merge(X, temp1, how='left', on='User_id')

    # 一共消费多少次
    X['u25'] = X.u2 + X.u5

    # 用户使用优惠券消费占比
    X['u20'] = X.u2 / X.u25

    # 正常消费平均间隔
    temp = pd.merge(temp, temp.groupby('User_id').Date.max().reset_index(name='max'))
    temp = pd.merge(temp, temp.groupby('User_id').Date.min().reset_index(name='min'))
    temp = pd.merge(temp, temp.groupby('User_id').size().reset_index(name='len'))
    temp['u6'] = ((temp['max'] - temp['min']).dt.days / (temp['len'] - 1))
    temp = temp.drop_duplicates('User_id')
    X = pd.merge(X, temp[['User_id', 'u6']], how='left', on='User_id')

    # 优惠券消费平均间隔
    temp = pd.merge(coupon_consume, user_coupon_consume.Date.max().reset_index(name='max'))
    temp = pd.merge(temp, temp.groupby('User_id').Date.min().reset_index(name='min'))
    temp = pd.merge(temp, temp.groupby('User_id').size().reset_index(name='len'))
    temp['u7'] = ((temp['max'] - temp['min']).dt.days / (temp['len'] - 1))
    temp = temp.drop_duplicates('User_id')
    X = pd.merge(X, temp[['User_id', 'u7']], how='left', on='User_id')

    # 15天内平均会普通消费几次
    X['u8'] = X.u6 / 15

    # 15天内平均会优惠券消费几次
    X['u9'] = X.u7 / 15

    # 领取优惠券到使用优惠券的平均间隔时间
    temp = coupon_consume.copy()
    temp['days'] = (temp.Date - temp.Date_received).dt.days
    temp = (temp.groupby('User_id').days.sum() / temp.groupby('User_id').size()).reset_index(name='u10')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 在15天内使用掉优惠券的值大小
    X['u11'] = X.u10 / 15

    # 领取优惠券到使用优惠券间隔小于15天的次数
    temp = coupon_consume.copy()
    temp['days'] = (temp.Date - temp.Date_received).dt.days
    temp = temp[temp.days <= 15]
    temp = temp.groupby('User_id').size().reset_index(name='u21')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户15天使用掉优惠券的次数除以使用优惠券的次数
    X['u22'] = X.u21 / X.u2

    # 用户15天使用掉优惠券的次数除以领取优惠券未消费的次数
    X['u23'] = X.u21 / X.u3

    # 用户15天使用掉优惠券的次数除以领取优惠券的总次数
    X['u24'] = X.u21 / X.u1

    # 消费优惠券的平均折率
    temp = user_coupon_consume.discount_rate.mean().reset_index(name='u45')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销优惠券的最低消费折率
    temp = user_coupon_consume.discount_rate.min().reset_index(name='u27')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销优惠券的最高消费折率
    temp = user_coupon_consume.discount_rate.max().reset_index(name='u28')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销过的不同优惠券数量
    temp = coupon_consume.groupby(['User_id', 'Coupon_id']).size()
    temp = temp.groupby('User_id').size().reset_index(name='u32')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户领取所有不同优惠券数量
    temp = offline[offline.Date_received != date_null]
    temp = temp.groupby(['User_id', 'Coupon_id']).size().reset_index(name='u47')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Coupon_id'])

    # 用户核销过的不同优惠券数量占所有不同优惠券的比重
    X['u33'] = X.u32 / X.u47

    # 用户平均每种优惠券核销多少张
    X['u34'] = X.u2 / X.u47

    # 核销优惠券用户-商家平均距离
    temp = offline[(offline.Coupon_id != 0) & (offline.Date != date_null) & (offline.Distance != 11)]
    temp = temp.groupby('User_id').Distance
    temp = pd.merge(temp.count().reset_index(name='x'), temp.sum().reset_index(name='y'), on='User_id')
    temp['u35'] = temp.y / temp.x
    temp = temp[['User_id', 'u35']]
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销优惠券中的最小用户-商家距离
    temp = coupon_consume[coupon_consume.Distance != 11]
    temp = temp.groupby('User_id').Distance.min().reset_index(name='u36')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销优惠券中的最大用户-商家距离
    temp = coupon_consume[coupon_consume.Distance != 11]
    temp = temp.groupby('User_id').Distance.max().reset_index(name='u37')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 优惠券类型
    discount_types = [
        '0.2', '0.5', '0.6', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '30:20', '50:30', '10:5',
        '20:10', '100:50', '200:100', '50:20', '30:10', '150:50', '100:30', '20:5', '200:50', '5:1',
        '50:10', '100:20', '150:30', '30:5', '300:50', '200:30', '150:20', '10:1', '50:5', '100:10',
        '200:20', '300:30', '150:10', '300:20', '500:30', '20:1', '100:5', '200:10', '30:1', '150:5',
        '300:10', '200:5', '50:1', '100:1',
    ]
    X['discount_type'] = -1
    for k, v in enumerate(discount_types):
        X.loc[X.Discount_rate == v, 'discount_type'] = k

    # 不同优惠券领取次数
    temp = offline.groupby(['User_id', 'Discount_rate']).size().reset_index(name='u41')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Discount_rate'])

    # 不同优惠券使用次数
    temp = coupon_consume.groupby(['User_id', 'Discount_rate']).size().reset_index(name='u42')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Discount_rate'])

    # 不同优惠券不使用次数
    temp = coupon_no_consume.groupby(['User_id', 'Discount_rate']).size().reset_index(name='u43')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Discount_rate'])

    # 不同打折优惠券使用率
    X['u44'] = X.u42 / X.u41

    # 满减类型优惠券领取次数
    temp = offline[offline.Discount_rate.str.contains(':') == True]
    temp = temp.groupby('User_id').size().reset_index(name='u48')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 打折类型优惠券领取次数
    temp = offline[offline.Discount_rate.str.contains('\.') == True]
    temp = temp.groupby('User_id').size().reset_index(name='u49')
    X = pd.merge(X, temp, how='left', on='User_id')

    '''offline merchant features'''

    # 商户消费次数
    temp = offline[offline.Date != date_null].groupby('Merchant_id').size().reset_index(name='m0')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券被领取后核销次数
    temp = coupon_consume.groupby('Merchant_id').size().reset_index(name='m1')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商户正常消费笔数
    X['m2'] = X.m0.fillna(0) - X.m1.fillna(0)

    # 商家优惠券被领取次数
    temp = offline[offline.Date_received != date_null].groupby('Merchant_id').size().reset_index(name='m3')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券被领取后核销率
    X['m4'] = X.m1 / X.m3

    # 商家优惠券被领取后不核销次数
    temp = coupon_no_consume.groupby('Merchant_id').size().reset_index(name='m7')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商户当天优惠券领取次数
    temp = X[X.Date_received != date_null]
    temp = temp.groupby(['Merchant_id', 'Date_received']).size().reset_index(name='m5')
    X = pd.merge(X, temp, how='left', on=['Merchant_id', 'Date_received'])

    # 商户当天优惠券领取人数
    temp = X[X.Date_received != date_null]
    temp = temp.groupby(['User_id', 'Merchant_id', 'Date_received']).size().reset_index()
    temp = temp.groupby(['Merchant_id', 'Date_received']).size().reset_index(name='m6')
    X = pd.merge(X, temp, how='left', on=['Merchant_id', 'Date_received'])

    # 商家优惠券核销的平均消费折率
    temp = coupon_consume.groupby('Merchant_id').discount_rate.mean().reset_index(name='m8')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券核销的最小消费折率
    temp = coupon_consume.groupby('Merchant_id').discount_rate.max().reset_index(name='m9')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券核销的最大消费折率
    temp = coupon_consume.groupby('Merchant_id').discount_rate.min().reset_index(name='m10')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券核销不同的用户数量
    temp = coupon_consume.groupby(['Merchant_id', 'User_id']).size()
    temp = temp.groupby('Merchant_id').size().reset_index(name='m11')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家优惠券领取不同的用户数量
    temp = offline[offline.Date_received != date_null].groupby(['Merchant_id', 'User_id']).size()
    temp = temp.groupby('Merchant_id').size().reset_index(name='m12')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 核销商家优惠券的不同用户数量其占领取不同的用户比重
    X['m13'] = X.m11 / X.m12

    # 商家优惠券平均每个用户核销多少张
    X['m14'] = X.m1 / X.m12

    # 商家被核销过的不同优惠券数量
    temp = coupon_consume.groupby(['Merchant_id', 'Coupon_id']).size()
    temp = temp.groupby('Merchant_id').size().reset_index(name='m15')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家领取过的不同优惠券数量的比重
    temp = offline[offline.Date_received != date_null].groupby(['Merchant_id', 'Coupon_id']).size()
    temp = temp.groupby('Merchant_id').count().reset_index(name='m18')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
    X['m19'] = X.m15 / X.m18

    # 商家被核销优惠券的平均时间
    temp = pd.merge(coupon_consume, coupon_consume.groupby('Merchant_id').Date.max().reset_index(name='max'))
    temp = pd.merge(temp, temp.groupby('Merchant_id').Date.min().reset_index(name='min'))
    temp = pd.merge(temp, temp.groupby('Merchant_id').size().reset_index(name='len'))
    temp['m20'] = ((temp['max'] - temp['min']).dt.days / (temp['len'] - 1))
    temp = temp.drop_duplicates('Merchant_id')
    X = pd.merge(X, temp[['Merchant_id', 'm20']], how='left', on='Merchant_id')

    # 商家被核销优惠券中的用户-商家平均距离
    temp = coupon_consume[coupon_consume.Distance != 11].groupby('Merchant_id').Distance
    temp = pd.merge(temp.count().reset_index(name='x'), temp.sum().reset_index(name='y'), on='Merchant_id')
    temp['m21'] = temp.y / temp.x
    temp = temp[['Merchant_id', 'm21']]
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家被核销优惠券中的用户-商家最小距离
    temp = coupon_consume[coupon_consume.Distance != 11]
    temp = temp.groupby('Merchant_id').Distance.min().reset_index(name='m22')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家被核销优惠券中的用户-商家最大距离
    temp = coupon_consume[coupon_consume.Distance != 11]
    temp = temp.groupby('Merchant_id').Distance.max().reset_index(name='m23')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    """offline coupon features"""

    # 此优惠券一共发行多少张
    temp = offline[offline.Coupon_id != 0].groupby('Coupon_id').size().reset_index(name='c1')
    X = pd.merge(X, temp, how='left', on='Coupon_id')

    # 此优惠券一共被使用多少张
    temp = coupon_consume.groupby('Coupon_id').size().reset_index(name='c2')
    X = pd.merge(X, temp, how='left', on='Coupon_id')

    # 优惠券使用率
    X['c3'] = X.c2 / X.c1

    # 没有使用的数目
    X['c4'] = X.c1 - X.c2

    # 此优惠券在当天发行了多少张
    temp = X.groupby(['Coupon_id', 'Date_received']).size().reset_index(name='c5')
    X = pd.merge(X, temp, how='left', on=['Coupon_id', 'Date_received'])

    # 优惠券类型(直接优惠为0, 满减为1)
    X['c6'] = 0
    X.loc[X.Discount_rate.str.contains(':') == True, 'c6'] = 1

    # 不同打折优惠券领取次数
    temp = offline.groupby('Discount_rate').size().reset_index(name='c8')
    X = pd.merge(X, temp, how='left', on='Discount_rate')

    # 不同打折优惠券使用次数
    temp = coupon_consume.groupby('Discount_rate').size().reset_index(name='c9')
    X = pd.merge(X, temp, how='left', on='Discount_rate')

    # 不同打折优惠券不使用次数
    temp = coupon_no_consume.groupby('Discount_rate').size().reset_index(name='c10')
    X = pd.merge(X, temp, how='left', on='Discount_rate')

    # 不同打折优惠券使用率
    X['c11'] = X.c9 / X.c8

    # 优惠券核销平均时间
    temp = pd.merge(coupon_consume, coupon_consume.groupby('Coupon_id').Date.max().reset_index(name='max'))
    temp = pd.merge(temp, temp.groupby('Coupon_id').Date.min().reset_index(name='min'))
    temp = pd.merge(temp, temp.groupby('Coupon_id').size().reset_index(name='count'))
    temp['c12'] = ((temp['max'] - temp['min']).dt.days / (temp['count'] - 1))
    temp = temp.drop_duplicates('Coupon_id')
    X = pd.merge(X, temp[['Coupon_id', 'c12']], how='left', on='Coupon_id')

    '''user merchant feature'''

    # 用户领取商家的优惠券次数
    temp = offline[offline.Coupon_id != 0]
    temp = temp.groupby(['User_id', 'Merchant_id']).size().reset_index(name='um1')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户领取商家的优惠券后不核销次数
    temp = coupon_no_consume.groupby(['User_id', 'Merchant_id']).size().reset_index(name='um2')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户领取商家的优惠券后核销次数
    temp = coupon_consume.groupby(['User_id', 'Merchant_id']).size().reset_index(name='um3')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户领取商家的优惠券后核销率
    X['um4'] = X.um3 / X.um1

    # 用户对每个商家的不核销次数占用户总的不核销次数的比重
    temp = coupon_no_consume.groupby('User_id').size().reset_index(name='temp')
    X = pd.merge(X, temp, how='left', on='User_id')
    X['um5'] = X.um2 / X.temp
    X.drop(columns='temp', inplace=True)

    # 用户在商店总共消费过几次
    temp = offline[offline.Date != date_null].groupby(['User_id', 'Merchant_id']).size().reset_index(name='um6')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户在商店普通消费次数
    temp = offline[(offline.Coupon_id == 0) & (offline.Date != date_null)]
    temp = temp.groupby(['User_id', 'Merchant_id']).size().reset_index(name='um7')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户当天在此商店领取的优惠券数目
    temp = offline[offline.Date_received != date_null]
    temp = temp.groupby(['User_id', 'Merchant_id', 'Date_received']).size().reset_index(name='um8')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id', 'Date_received'])

    # 用户领取优惠券不同商家数量
    temp = offline[offline.Coupon_id == offline.Coupon_id]
    temp = temp.groupby(['User_id', 'Merchant_id']).size().reset_index()
    temp = temp.groupby('User_id').size().reset_index(name='um9')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销优惠券不同商家数量
    temp = coupon_consume.groupby(['User_id', 'Merchant_id']).size()
    temp = temp.groupby('User_id').size().reset_index(name='um10')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户核销过优惠券的不同商家数量占所有不同商家的比重
    X['um11'] = X.um10 / X.um9

    # 用户平均核销每个商家多少张优惠券
    X['um12'] = X.u2 / X.um9

    '''other feature'''

    # 用户领取的所有优惠券数目
    temp = X.groupby('User_id').size().reset_index(name='o1')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户领取的特定优惠券数目
    temp = X.groupby(['User_id', 'Coupon_id']).size().reset_index(name='o2')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Coupon_id'])

    # multiple threads
    # data split
    stop = len(X)
    global cpu_jobs
    step = int(ceil(stop / cpu_jobs))

    X_chunks = [X[i:i + step] for i in range(0, stop, step)]
    X_list = [X] * cpu_jobs
    counters = [i for i in range(cpu_jobs)]

    start = datetime.datetime.now()
    with ProcessPoolExecutor() as e:
        X = pd.concat(e.map(task, X_chunks, X_list, counters))
        print('time:', str(datetime.datetime.now() - start).split('.')[0])
    # multiple threads

    # 用户领取优惠券平均时间间隔
    temp = pd.merge(X, X.groupby('User_id').Date_received.max().reset_index(name='max'))
    temp = pd.merge(temp, temp.groupby('User_id').Date_received.min().reset_index(name='min'))
    temp = pd.merge(temp, temp.groupby('User_id').size().reset_index(name='len'))
    temp['o7'] = ((temp['max'] - temp['min']).dt.days / (temp['len'] - 1))
    temp = temp.drop_duplicates('User_id')
    X = pd.merge(X, temp[['User_id', 'o7']], how='left', on='User_id')

    # 用户领取特定商家的优惠券数目
    temp = X.groupby(['User_id', 'Merchant_id']).size().reset_index(name='o8')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Merchant_id'])

    # 用户领取的不同商家数目
    temp = X.groupby(['User_id', 'Merchant_id']).size()
    temp = temp.groupby('User_id').size().reset_index(name='o9')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户当天领取的优惠券数目
    temp = X.groupby(['User_id', 'Date_received']).size().reset_index(name='o10')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Date_received'])

    # 用户当天领取的特定优惠券数目
    temp = X.groupby(['User_id', 'Coupon_id', 'Date_received']).size().reset_index(name='o11')
    X = pd.merge(X, temp, how='left', on=['User_id', 'Coupon_id', 'Date_received'])

    # 用户领取的所有优惠券种类数目
    temp = X.groupby(['User_id', 'Coupon_id']).size()
    temp = temp.groupby('User_id').size().reset_index(name='o12')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 商家被领取的优惠券数目
    temp = X.groupby('Merchant_id').size().reset_index(name='o13')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家被领取的特定优惠券数目
    temp = X.groupby(['Merchant_id', 'Coupon_id']).size().reset_index(name='o14')
    X = pd.merge(X, temp, how='left', on=['Merchant_id', 'Coupon_id'])

    # 商家被多少不同用户领取的数目
    temp = X.groupby(['Merchant_id', 'User_id']).size()
    temp = temp.groupby('Merchant_id').size().reset_index(name='o15')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    # 商家发行的所有优惠券种类数目
    temp = X.groupby(['Merchant_id', 'Coupon_id']).size()
    temp = temp.groupby('Merchant_id').size().reset_index(name='o16')
    X = pd.merge(X, temp, how='left', on='Merchant_id')

    print("After offline feature process: ", len(X), len(X.columns))
    print("X.head(5)", X.head(5))
    print("offline.head(5)", offline.head(5))

    return X

# 线上特征： 线上数据较多，且没有相同的商户、优惠券，所以首先做数据的筛选，将线下没有出现过的用户数据剔除掉
# 观察数据后发现： 当月特征： 观察发现当用户领了某券消费后，大部分商店会在使用券的当天再发一张券,所以可以假定该用户领了该券的个数越多，用户使用的概率越大。
# 另外如果某条记录是该用户最后一次领该券，很可能是因为这次没有使用所以没有再给该用户发券，则该记录不使用券的概率就很大
def get_online_features(online, X):
    # temp = online[online.Coupon_id == online.Coupon_id]
    # coupon_consume = temp[temp.Date == temp.Date]
    # coupon_no_consume = temp[temp.Date != temp.Date]

    # 用户线上操作次数
    temp = online.groupby('User_id').size().reset_index(name='on_u1')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上点击次数
    temp = online[online.Action == 0].groupby('User_id').size().reset_index(name='on_u2')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上点击率
    X['on_u3'] = X.on_u2 / X.on_u1

    # 用户线上购买次数
    temp = online[online.Action == 1].groupby('User_id').size().reset_index(name='on_u4')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上购买率
    X['on_u5'] = X.on_u4 / X.on_u1

    # 用户线上领取次数
    temp = online[online.Coupon_id != 0].groupby('User_id').size().reset_index(name='on_u6')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上领取率
    X['on_u7'] = X.on_u6 / X.on_u1

    # 用户线上不消费次数
    temp = online[(online.Date == date_null) & (online.Coupon_id != 0)]
    temp = temp.groupby('User_id').size().reset_index(name='on_u8')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上优惠券核销次数
    temp = online[(online.Date != date_null) & (online.Coupon_id != 0)]
    temp = temp.groupby('User_id').size().reset_index(name='on_u9')
    X = pd.merge(X, temp, how='left', on='User_id')

    # 用户线上优惠券核销率
    X['on_u10'] = X.on_u9 / X.on_u6

    # 用户线下不消费次数占线上线下总的不消费次数的比重
    X['on_u11'] = X.u3 / (X.on_u8 + X.u3)

    # 用户线下的优惠券核销次数占线上线下总的优惠券核销次数的比重
    X['on_u12'] = X.u2 / (X.on_u9 + X.u2)

    # 用户线下领取的记录数量占总的记录数量的比重
    X['on_u13'] = X.u1 / (X.on_u6 + X.u1)

    # # 消费优惠券的平均折率
    # temp = coupon_consume.groupby('User_id').discount_rate.mean().reset_index(name='ou14')
    # X = pd.merge(X, temp, how='left', on='User_id')
    #
    # # 用户核销优惠券的最低消费折率
    # temp = coupon_consume.groupby('User_id').discount_rate.min().reset_index(name='ou15')
    # X = pd.merge(X, temp, how='left', on='User_id')
    #
    # # 用户核销优惠券的最高消费折率
    # temp = coupon_consume.groupby('User_id').discount_rate.max().reset_index(name='ou16')
    # X = pd.merge(X, temp, how='left', on='User_id')
    #
    # # 不同打折优惠券领取次数
    # temp = online.groupby('Discount_rate').size().reset_index(name='oc1')
    # X = pd.merge(X, temp, how='left', on='Discount_rate')
    #
    # # 不同打折优惠券使用次数
    # temp = coupon_consume.groupby('Discount_rate').size().reset_index(name='oc2')
    # X = pd.merge(X, temp, how='left', on='Discount_rate')
    #
    # # 不同打折优惠券不使用次数
    # temp = coupon_no_consume.groupby('Discount_rate').size().reset_index(name='oc3')
    # X = pd.merge(X, temp, how='left', on='Discount_rate')
    #
    # # 不同打折优惠券使用率
    # X['oc4'] = X.oc2 / X.oc1

    print("After online feature process: ", len(X), len(X.columns))
    print('----------\n')

    return X


def task(X_chunk, X, counter):
    print(counter, end=',', flush=True)
    X_chunk = X_chunk.copy()

    X_chunk['o17'] = -1
    X_chunk['o18'] = -1

    for i, user in X_chunk.iterrows():
        temp = X[X.User_id == user.User_id]

        temp1 = temp[temp.Date_received < user.Date_received]
        temp2 = temp[temp.Date_received > user.Date_received]

        # 用户此次之后/前领取的所有优惠券数目
        X_chunk.loc[i, 'o3'] = len(temp1)
        X_chunk.loc[i, 'o4'] = len(temp2)

        # 用户此次之后/前领取的特定优惠券数目
        X_chunk.loc[i, 'o5'] = len(temp1[temp1.Coupon_id == user.Coupon_id])
        X_chunk.loc[i, 'o6'] = len(temp2[temp2.Coupon_id == user.Coupon_id])

        # 用户上/下一次领取的时间间隔
        temp1 = temp1.sort_values(by='Date_received', ascending=False)
        if len(temp1):
            X_chunk.loc[i, 'o17'] = (user.Date_received - temp1.iloc[0].Date_received).days

        temp2 = temp2.sort_values(by='Date_received')
        if len(temp2):
            X_chunk.loc[i, 'o18'] = (temp2.iloc[0].Date_received - user.Date_received).days

    return X_chunk


def analysis():
    global date_null

    offline, online = get_preprocess_data()

    # Checking for missing data
    # NAs = pd.concat([data.isnull().sum()], axis=1, keys=['Train'])
    # NAs[NAs.sum(axis=1) > 0]

    t = offline.groupby('Distance').size().reset_index(name='receive_count')
    t1 = offline[(offline.Coupon_id != 0) & (offline.Date != date_null)]
    t1 = t1.groupby('Distance').size().reset_index(name='consume_count')
    t = pd.merge(t, t1, on='Distance')
    t['consume_rate'] = t.consume_count / t.receive_count  # 消费率

    t.to_csv('note.csv')

    # plt.bar(temp.Discount_rate.values, temp.total.values)
    # plt.bar(range(num), y1, bottom=y2, fc='r')
    # plt.show()

    print('有优惠券，购买商品条数', offline[(offline['Date_received'] != 'null') & (offline['Date'] != 'null')].shape[0])
    print('无优惠券，购买商品条数', offline[(offline['Date_received'] == 'null') & (offline['Date'] != 'null')].shape[0])
    print('有优惠券，不购买商品条数', offline[(offline['Date_received'] != 'null') & (offline['Date'] == 'null')].shape[0])
    print('无优惠券，不购买商品条数', offline[(offline['Date_received'] == 'null') & (offline['Date'] == 'null')].shape[0])

    # 在测试集中出现的用户但训练集没有出现
    # print('1. User_id in training set but not in test set', set(dftest['User_id']) - set(offline['User_id']))
    # 在测试集中出现的商户但训练集没有出现
    # print('2. Merchant_id in training set but not in test set', set(dftest['Merchant_id']) - set(offline['Merchant_id']))

    print('Discount_rate 类型:', offline['Discount_rate'].unique())
    print('Distance 类型:', offline['Distance'].unique())

    date_received = offline['Date_received'].unique()
    date_received = sorted(date_received[date_received != 'null'])

    date_buy = offline['Date'].unique()
    date_buy = sorted(date_buy[date_buy != 'null'])

    date_buy = sorted(offline[offline['Date'] != 'null']['Date'])
    print('优惠券收到日期从', date_received[0], '到', date_received[-1])
    print('消费日期从', date_buy[0], '到', date_buy[-1])

    # 看一下每天的顾客收到coupon的数目，以及收到coupon后用coupon消费的数目。见analysis1.png
    couponbydate = offline[offline['Date_received'] != 'null'][['Date_received', 'Date']].groupby(['Date_received'],
                                                                                              as_index=False).count()
    couponbydate.columns = ['Date_received', 'count']
    buybydate = offline[(offline['Date'] != 'null') & (offline['Date_received'] != 'null')][
        ['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
    buybydate.columns = ['Date_received', 'count']

    sns.set_style('ticks')
    sns.set_context("notebook", font_scale=1.4)
    plt.figure(figsize=(12, 8))
    date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')

    plt.subplot(211)
    plt.bar(date_received_dt, couponbydate['count'], label='number of coupon received')
    plt.bar(date_received_dt, buybydate['count'], label='number of coupon used')
    plt.yscale('log')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(212)
    plt.bar(date_received_dt, buybydate['count'] / couponbydate['count'])
    plt.ylabel('Ratio(coupon used/coupon received)')
    plt.tight_layout()

    exit()

    '''有优惠券，购买商品条数 75382

        无优惠券，购买商品条数 701602

        有优惠券，不购买商品条数 977900

        无优惠券，不购买商品条数 0
        
        1. User_id in training set but not in test set {2495873, 1286474}

        2. Merchant_id in training set but not in test set {5920}
        
        Discount_rate 类型: ['null' '150:20' '20:1' '200:20' '30:5' '50:10' '10:5' '100:10' '200:30'

         '20:5' '30:10' '50:5' '150:10' '100:30' '200:50' '100:50' '300:30'
        
         '50:20' '0.9' '10:1' '30:1' '0.95' '100:5' '5:1' '100:20' '0.8' '50:1'
        
         '200:10' '300:20' '100:1' '150:30' '300:50' '20:10' '0.85' '0.6' '150:50'
        
         '0.75' '0.5' '200:5' '0.7' '30:20' '300:10' '0.2' '50:30' '200:100'
        
         '150:5']
        
        Distance 类型: ['0' '1' 'null' '2' '10' '4' '7' '9' '3' '5' '6' '8']
        
        优惠券收到日期从 20160101 到 20160615

        消费日期从 20160101 到 20160630
'''

'''
新建关于星期的特征

def getWeekday(row):
    if row == 'null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1
        
def change_weekday:
    dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
    dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

    # weekday_type :  周六和周日为1，其他为0
    dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)
    dftest['weekday_type'] = dftest['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)
    
    # change weekday to one-hot encoding 
    weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
    print(weekdaycols)
    
    tmpdf = pd.get_dummies(dfoff['weekday'].replace('null', np.nan))
    tmpdf.columns = weekdaycols
    dfoff[weekdaycols] = tmpdf
    
    tmpdf = pd.get_dummies(dftest['weekday'].replace('null', np.nan))
    tmpdf.columns = weekdaycols
    dftest[weekdaycols] = tmpdf
    
weekday : {null, 1, 2, 3, 4, 5, 6, 7}
weekday_type : {1, 0}（周六和周日为1，其他为0）
Weekda_1 : {1, 0, 0, 0, 0, 0, 0}
Weekday_2 : {0, 1, 0, 0, 0, 0, 0}
Weekday_3 : {0, 0, 1, 0, 0, 0, 0}
Weekday_4 : {0, 0, 0, 1, 0, 0, 0}
Weekday_5 : {0, 0, 0, 0, 1, 0, 0}
Weekday_6 : {0, 0, 0, 0, 0, 1, 0}
Weekday_7 : {0, 0, 0, 0, 0, 0, 1}

'''

def detect_duplicate_columns():
    X = get_train_data()
    X = X[:1000]

    for index1 in range(len(X.columns) - 1):
        for index2 in range(index1 + 1, len(X.columns)):
            column1 = X.columns[index1]
            column2 = X.columns[index2]
            X[column1] = X[column1].astype(str)
            X[column2] = X[column2].astype(str)
            temp = len(X[X[column1] == X[column2]])
            if temp == len(X):
                print(column1, column2, temp)
    exit()


def feature_importance_score():
    clf = model.train_xgb()
    fscores = pd.Series(clf.get_booster().get_fscore()).sort_values(ascending=False)
    fscores.plot(kind='bar', title='Feature Importance', figsize=(20,10))
    plt.ylabel('Feature Importance Score')
    plt.show()
    # exit()

def feature_selection():
    data = get_train_data()

    train_data, test_data = train_test_split(data,
                                             train_size=100000,
                                             random_state=0
                                             )

    X = train_data.copy().drop(columns='Coupon_id')
    y = X.pop('label')

    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X = sel.fit_transform(X)
    print(X.shape)
    # Create the RFE object and rank each pixel

if __name__ == '__main__':
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 200)

    start = datetime.datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    # Logger = log.Logger(start)
    # log = Logger.get_logger()
    logger.init_logger(start)
    cpu_jobs = os.cpu_count() - 1
    date_null = pd.to_datetime('1970-01-01', format='%Y-%m-%d')

    analysis()
    get_train_data()
    # detect_duplicate_columns()
    # feature_importance_score()

    # grid_search_gbdt()

    # train_gbdt()
    # predict('gbdt')

    # grid_search_xgb()
    # train_xgb()
    # predict('xgb')

    # grid_search_lgb()
    # train_lgb()
    # predict('lgb')

    # grid_search_cat()
    # train_cat()
    # predict('cat')

    # grid_search_rf()
    # train_rf_gini()
    # predict('rf_gini')

    # grid_search_rf('entropy')
    # train_rf_entropy()
    # predict('rf_entropy')

    # grid_search_et()
    # train_et_gini()
    # predict('et_gini')

    # grid_search_et('entropy')
    # train_et_entropy()
    # predict('et_entropy')

    # blending()
    # predict('blending')

    log = logger.get_logger()
    log += 'time: %s\n' % str((datetime.datetime.now() - start)).split('.')[0]
    log += '----------------------------------------------------\n'
    open('%s.log' % os.path.basename(__file__), 'a').write(log)
    print(log)

'''
2019-10-06 11:36:04
132943 11
X.head(5)     User_id  Merchant_id  Coupon_id Discount_rate  Distance Date_received       Date  discount_rate_x  discount_rate_y  discount_rate  label
7   1832624         3381       7610        200:20         0    2016-04-29 1970-01-01            200.0             20.0       0.900000      0
18   163606         1569       5054        200:30        10    2016-04-21 1970-01-01            200.0             30.0       0.850000      0
33  1113008         1361      11166          20:1         0    2016-05-15 2016-05-21             20.0              1.0       0.950000      1
43  4061024         3381       7610        200:20        10    2016-04-26 1970-01-01            200.0             20.0       0.900000      0
52   106443          450       3732          30:5        11    2016-04-29 1970-01-01             30.0              5.0       0.833333      0
offline.head(5)    User_id  Merchant_id  Coupon_id Discount_rate  Distance Date_received       Date  discount_rate_x  discount_rate_y  discount_rate
0  1439408         2632          0           NaN         0    1970-01-01 2016-02-17              NaN              NaN            NaN
2  1439408         2632       8591          20:1         0    2016-02-17 1970-01-01             20.0              1.0           0.95
3  1439408         2632       1078          20:1         0    2016-03-19 1970-01-01             20.0              1.0           0.95
5  1439408         2632          0           NaN         0    1970-01-01 2016-05-16              NaN              NaN            NaN
8  2029232         3381      11951        200:20         1    2016-01-29 1970-01-01            200.0             20.0           0.90
0,1,2,time: 0:17:32
132943 111
X.head(5)    User_id  Merchant_id  Coupon_id Discount_rate  Distance Date_received       Date  discount_rate_x  discount_rate_y  discount_rate  label  weekday  day  u2   u3  u19   u1  u4   u5  u25  u20    u6  u7   u8  u9  u10  u11  u21  u22  u23  u24  u45  u27  u28  u32  u47  u33  u34  u35  u36  u37  discount_type  u41  u42  u43  u44  u48  u49       m0      m1       m2       m3        m4       m7    m5    m6        m8   m9       m10     m11      m12       m13       m14  m15  m18       m19       m20       m21  m22   m23      c1     c2        c3      c4    c5  c6        c8       c9       c10       c11       c12  um1  um2  um3  um4  um5  um6  um7  um8  um9  um10  um11  um12  o1  o2  o17  o18   o3   o4   o5   o6  o7  o8  o9  o10  o11  o12    o13    o14    o15  o16
0  1832624         3381       7610        200:20         0    2016-04-29 1970-01-01            200.0             20.0       0.900000      0        4   29 NaN  NaN  NaN  0.0 NaN  NaN  NaN  NaN   NaN NaN  NaN NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN             32  NaN  NaN  NaN  NaN  NaN  NaN  34494.0  2200.0  32294.0  83862.0  0.026234  81662.0  1293  1293  0.723818  0.9  0.500000  1978.0  69639.0  0.028404  0.031591  8.0  8.0  1.000000  0.049113  1.643602  0.0  10.0  7757.0  129.0  0.016630  7628.0  1293   1   65391.0   1151.0   64240.0  0.017602  0.343750  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   NaN   NaN   NaN   1   1   -1   -1  0.0  0.0  0.0  0.0 NaN   1   1    1    1    1  19623  19623  19613    1
1   163606         1569       5054        200:30        10    2016-04-21 1970-01-01            200.0             30.0       0.850000      0        3   21 NaN  NaN  NaN  0.0 NaN  NaN  NaN  NaN   NaN NaN  NaN NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN             27  NaN  NaN  NaN  NaN  NaN  NaN   1712.0   107.0   1605.0  12799.0  0.008360  12692.0   765   765  0.778816  0.9  0.700000    86.0  12296.0  0.006994  0.008702  7.0  7.0  1.000000  1.141509  2.500000  0.0  10.0   827.0    4.0  0.004837   823.0   764   1    6130.0    205.0    5925.0  0.033442  6.666667  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   NaN   NaN   NaN   1   1   -1   -1  0.0  0.0  0.0  0.0 NaN   1   1    1    1    1  10620  10543  10596    4
2  1113008         1361      11166          20:1         0    2016-05-15 2016-05-21             20.0              1.0       0.950000      1        6   15 NaN  1.0  NaN  1.0 NaN  3.0  NaN  NaN  72.0 NaN  4.8 NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN             37  NaN  NaN  NaN  NaN  1.0  NaN     84.0     NaN     84.0      NaN       NaN      NaN     1     1       NaN  NaN       NaN     NaN      NaN       NaN       NaN  NaN  NaN       NaN       NaN       NaN  NaN   NaN     NaN    NaN       NaN     NaN     1   1    5994.0    654.0    5340.0  0.109109       NaN  NaN  NaN  NaN  NaN  NaN  1.0  1.0  NaN  2.0   NaN   NaN   NaN   1   1   -1   -1  0.0  0.0  0.0  0.0 NaN   1   1    1    1    1      3      3      3    1
3  4061024         3381       7610        200:20        10    2016-04-26 1970-01-01            200.0             20.0       0.900000      0        1   26 NaN  2.0  NaN  2.0 NaN  1.0  NaN  NaN   NaN NaN  NaN NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN             32  NaN  NaN  NaN  NaN  2.0  NaN  34494.0  2200.0  32294.0  83862.0  0.026234  81662.0  1295  1295  0.723818  0.9  0.500000  1978.0  69639.0  0.028404  0.031591  8.0  8.0  1.000000  0.049113  1.643602  0.0  10.0  7757.0  129.0  0.016630  7628.0  1295   1   65391.0   1151.0   64240.0  0.017602  0.343750  1.0  1.0  NaN  NaN  0.5  NaN  NaN  NaN  3.0   NaN   NaN   NaN   1   1   -1   -1  0.0  0.0  0.0  0.0 NaN   1   1    1    1    1  19623  19623  19613    1
4   106443          450       3732          30:5        11    2016-04-29 1970-01-01             30.0              5.0       0.833333      0        4   29 NaN  1.0  NaN  1.0 NaN  NaN  NaN  NaN   NaN NaN  NaN NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN             25  NaN  NaN  NaN  NaN  1.0  NaN  17468.0   386.0  17082.0  28285.0  0.013647  27899.0   631   631  0.812090  0.9  0.666667   318.0  26993.0  0.011781  0.014300  6.0  7.0  0.857143  0.327273  0.816438  0.0  10.0   635.0   63.0  0.099213   572.0   631   1  135211.0  11989.0  123222.0  0.088669  0.709677  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  1.0   NaN   NaN   NaN   1   1   -1   -1  0.0  0.0  0.0  0.0 NaN   1   1    1    1    1  10303   8037   9926    5
offline.head(5)    User_id  Merchant_id  Coupon_id Discount_rate  Distance Date_received       Date  discount_rate_x  discount_rate_y  discount_rate
0  1439408         2632          0           NaN         0    1970-01-01 2016-02-17              NaN              NaN            NaN
2  1439408         2632       8591          20:1         0    2016-02-17 1970-01-01             20.0              1.0           0.95
3  1439408         2632       1078          20:1         0    2016-03-19 1970-01-01             20.0              1.0           0.95
5  1439408         2632          0           NaN         0    1970-01-01 2016-05-16              NaN              NaN            NaN
8  2029232         3381      11951        200:20         1    2016-01-29 1970-01-01            200.0             20.0           0.90
132943 124
----------

252586 11
X.head(5)     User_id  Merchant_id  Coupon_id Discount_rate  Distance Date_received       Date  discount_rate_x  discount_rate_y  discount_rate  label
1   1439408         4663      11002        150:20         1    2016-05-28 1970-01-01            150.0             20.0       0.866667      0
4   1439408         2632       8591          20:1         0    2016-06-13 1970-01-01             20.0              1.0       0.950000      0
6   1439408         2632       8591          20:1         0    2016-05-16 2016-06-13             20.0              1.0       0.950000      0
9   2029232          450       1532          30:5         0    2016-05-30 1970-01-01             30.0              5.0       0.833333      0
10  2029232         6459      12737          20:1         0    2016-05-19 1970-01-01             20.0              1.0       0.950000      0
offline.head(5)     User_id  Merchant_id  Coupon_id Discount_rate  Distance Date_received       Date  discount_rate_x  discount_rate_y  discount_rate
2   1439408         2632       8591          20:1         0    2016-02-17 1970-01-01             20.0              1.0           0.95
3   1439408         2632       1078          20:1         0    2016-03-19 1970-01-01             20.0              1.0           0.95
7   1832624         3381       7610        200:20         0    2016-04-29 1970-01-01            200.0             20.0           0.90
17    73611         2099      12034        100:10        11    2016-02-07 1970-01-01            100.0             10.0           0.90
18   163606         1569       5054        200:30        10    2016-04-21 1970-01-01            200.0             30.0           0.85
0,1,2,time: 0:23:10
252586 111
X.head(5)    User_id  Merchant_id  Coupon_id Discount_rate  Distance Date_received       Date  discount_rate_x  discount_rate_y  discount_rate  label  weekday  day  u2   u3  u19   u1  u4   u5  u25  u20  u6  u7  u8  u9  u10  u11  u21  u22  u23  u24  u45  u27  u28  u32  u47  u33  u34  u35  u36  u37  discount_type  u41  u42  u43  u44  u48  u49      m0     m1      m2       m3        m4       m7    m5    m6        m8        m9       m10    m11      m12       m13       m14  m15  m18       m19        m20       m21  m22   m23    c1   c2    c3    c4    c5  c6        c8      c9       c10       c11   c12  um1  um2  um3  um4  um5  um6  um7  um8  um9  um10  um11  um12  o1  o2  o17  o18   o3   o4   o5   o6    o7  o8  o9  o10  o11  o12    o13    o14    o15  o16
0  1439408         4663      11002        150:20         1    2016-05-28 1970-01-01            150.0             20.0       0.866667      0        5   28 NaN  2.0  NaN  2.0 NaN  1.0  NaN  NaN NaN NaN NaN NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN             28  NaN  NaN  NaN  NaN  2.0  NaN   650.0   16.0   634.0    407.0  0.039312    391.0  7861  7855  0.866667  0.866667  0.866667   16.0    405.0  0.039506  0.039506  2.0  4.0  0.500000   5.266667  2.200000  0.0   9.0   NaN  NaN   NaN   NaN  7726   1     939.0    56.0     883.0  0.059638   NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  1.0   NaN   NaN   NaN   3   1   12   16  1.0  1.0  0.0  0.0  14.0   1   2    1    1    2  10831   7730  10771    2
1  1439408         2632       8591          20:1         0    2016-06-13 1970-01-01             20.0              1.0       0.950000      0        0   13 NaN  2.0  NaN  2.0 NaN  1.0  NaN  NaN NaN NaN NaN NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  1.0  NaN  NaN  NaN  NaN  NaN             37  2.0  NaN  2.0  NaN  2.0  NaN    14.0    3.0    11.0     28.0  0.107143     25.0     1     1  0.950000  0.950000  0.950000    1.0     10.0  0.100000  0.300000  1.0  2.0  0.500000  28.000000  1.000000  1.0   1.0  20.0  3.0  0.15  17.0     1   1   13354.0  1727.0   11627.0  0.129325  28.0  2.0  2.0  NaN  NaN  1.0  1.0  1.0  NaN  1.0   NaN   NaN   NaN   3   2   16   -1  2.0  0.0  1.0  0.0  14.0   2   2    1    1    2      8      5      7    2
2  1439408         2632       8591          20:1         0    2016-05-16 2016-06-13             20.0              1.0       0.950000      0        0   16 NaN  2.0  NaN  2.0 NaN  1.0  NaN  NaN NaN NaN NaN NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  1.0  NaN  NaN  NaN  NaN  NaN             37  2.0  NaN  2.0  NaN  2.0  NaN    14.0    3.0    11.0     28.0  0.107143     25.0     1     1  0.950000  0.950000  0.950000    1.0     10.0  0.100000  0.300000  1.0  2.0  0.500000  28.000000  1.000000  1.0   1.0  20.0  3.0  0.15  17.0     1   1   13354.0  1727.0   11627.0  0.129325  28.0  2.0  2.0  NaN  NaN  1.0  1.0  1.0  NaN  1.0   NaN   NaN   NaN   3   2   -1   12  0.0  2.0  0.0  1.0  14.0   2   2    1    1    2      8      5      7    2
3  2029232          450       1532          30:5         0    2016-05-30 1970-01-01             30.0              5.0       0.833333      0        0   30 NaN  NaN  NaN  0.0 NaN  NaN  NaN  NaN NaN NaN NaN NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN             25  NaN  NaN  NaN  NaN  NaN  NaN  6235.0  498.0  5737.0  22465.0  0.022168  21967.0   284   284  0.817604  0.833333  0.750000  420.0  21496.0  0.019539  0.023167  5.0  7.0  0.714286   0.195171  0.866242  0.0  10.0   1.0  1.0  1.00   0.0   257   1  117025.0  9792.0  107233.0  0.083674   NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   NaN   NaN   NaN   2   1   11   -1  1.0  0.0  0.0  0.0  11.0   1   2    1    1    2  19928  11428  18072   10
4  2029232         6459      12737          20:1         0    2016-05-19 1970-01-01             20.0              1.0       0.950000      0        3   19 NaN  NaN  NaN  0.0 NaN  NaN  NaN  NaN NaN NaN NaN NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN             37  NaN  NaN  NaN  NaN  NaN  NaN     5.0    NaN     5.0      NaN       NaN      NaN     3     3       NaN       NaN       NaN    NaN      NaN       NaN       NaN  NaN  NaN       NaN        NaN       NaN  NaN   NaN   NaN  NaN   NaN   NaN     3   1   13354.0  1727.0   11627.0  0.129325   NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN   NaN   NaN   NaN   2   1   -1   11  0.0  1.0  0.0  0.0  11.0   1   2    1    1    2     16     16     15    1
offline.head(5)     User_id  Merchant_id  Coupon_id Discount_rate  Distance Date_received       Date  discount_rate_x  discount_rate_y  discount_rate
2   1439408         2632       8591          20:1         0    2016-02-17 1970-01-01             20.0              1.0           0.95
3   1439408         2632       1078          20:1         0    2016-03-19 1970-01-01             20.0              1.0           0.95
7   1832624         3381       7610        200:20         0    2016-04-29 1970-01-01            200.0             20.0           0.90
17    73611         2099      12034        100:10        11    2016-02-07 1970-01-01            100.0             10.0           0.90
18   163606         1569       5054        200:30        10    2016-04-21 1970-01-01            200.0             30.0           0.85
252586 124
----------
2019-10-06 11:36:04
time: 0:43:12
----------------------------------------------------
'''
