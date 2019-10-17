#!/usr/bin/python
# -*- coding: utf-8 -*-

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
import numpy as np
from xgboost.sklearn import XGBClassifier
from tianchi_o2o import feature
from tianchi_o2o import logger

cpu_jobs = os.cpu_count() - 1
log = logger.get_logger()


def fit_eval_metric(estimator, X, y, name=None):
    if name is None:
        name = estimator.__class__.__name__

    if name is 'XGBClassifier' or name is 'LGBMClassifier':
        estimator.fit(X, y, eval_metric='auc')
    else:
        estimator.fit(X, y)

    return estimator


def grid_search(estimator, param_grid):
    start = datetime.datetime.now()

    print('-----------search single param begin-----------')
    # print(start.strftime('%Y-%m-%d %H:%M:%S'))
    print(param_grid)
    print()

    data = feature.get_train_data()

    data, _ = train_test_split(data, train_size=10000, random_state=0)

    X = data.copy().drop(columns='Coupon_id')
    y = X.pop('label')

    estimator_name = estimator.__class__.__name__
    n_jobs = cpu_jobs
    if estimator_name is 'XGBClassifier' or estimator_name is 'LGBMClassifier' or estimator_name is 'CatBoostClassifier':
        n_jobs = 1

    clf = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='roc_auc', n_jobs=n_jobs,
                       cv=5
                       )

    clf = fit_eval_metric(clf, X, y, estimator_name)  # 拟合评估指标

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('%0.5f (+/-%0.05f) for %r' % (mean, std * 2, params))
    print()
    print('best params', clf.best_params_)
    print('best score', clf.best_score_)
    print('time: %s' % str((datetime.datetime.now() - start)).split('.')[0])
    print('-----------search single param end-----------')
    print()

    return clf.best_params_, clf.best_score_


# Pipeline 自动化 Grid Search，只要预先设定好使用的 Model 和参数的候选，就能自动搜索并记录最佳的 Model。
def grid_search_auto(steps, params, estimator):
    global log

    old_params = params.copy()

    print(params)

    while 1:
        print('---------------new grid search for all params------------------')
        # for循环调整各个参数达到局部最优
        for name, step in steps.items():
            score = 0

            start = params[name] - step['step']
            if start <= step['min']:
                start = step['min']

            stop = params[name] + step['step']
            if step['max'] != 'inf' and stop >= step['max']:
                stop = step['max']
            # 调整单个参数达到局部最优
            while 1:

                if str(step['step']).count('.') == 1:
                    stop += step['step'] / 10
                else:
                    stop += step['step']

                param_grid = {
                    # 最开始这里会产生3个搜索的参数，begin +- step，但之后的轮次只会根据方向每次搜索一个参数
                    name: np.arange(start, stop, step['step']),
                }

                best_params, best_score = grid_search(estimator.set_params(**params), param_grid)

                # 找到最优参数或者当前的得分best_score已经低于上一轮的score时结束
                if best_params[name] == params[name] or score > best_score:
                    print("got best params, round over:",estimator.__class__.__name__, params)
                    break

                # 当产生比当前结果更优的解时，则在此方向是继续寻找
                direction = (best_params[name] - params[name]) // abs(best_params[name] - params[name])  # 决定了每次只能跑两个参数
                start = stop = best_params[name] + step['step'] * direction # 根据方向让下一轮每次搜索一个参数

                score = best_score
                params[name] = best_params[name]

                if best_params[name] - step['step'] < step['min'] or (
                        step['max'] != 'inf' and best_params[name] + step['step'] > step['max']):
                    print("reach params limit, round over.", estimator.__class__.__name__, params)
                    break

                print("Next round search: ", estimator.__class__.__name__, params)

        # 最外层的while控制全局最优，任何一个参数发生了变化就要重新再调整，达到全局最优
        if old_params == params:
            break
        old_params = params

    print('grid search: %s\n%r\n' % (estimator.__class__.__name__, params))
    log += 'grid search: %s\n%r\n' % (estimator.__class__.__name__, params)
    logger.set_logger(log)


def grid_search_gbdt(get_param=False):
    params = {
        # 10
        'learning_rate': 1e-2,
        'n_estimators': 1900,
        'max_depth': 9,
        'min_samples_split': 200,
        'min_samples_leaf': 50,
        'subsample': .8,

        # 'learning_rate': 1e-1,
        # 'n_estimators': 101,
        # 'max_depth': 8,
        # 'min_samples_split': 200,
        # 'min_samples_leaf': 50,
        # 'subsample': .8,

        'random_state': 0
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 100, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 10, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 10, 'min': 1, 'max': 'inf'},
        'subsample': {'step': .1, 'min': .1, 'max': 1},
    }

    grid_search_auto(steps, params, GradientBoostingClassifier())


def grid_search_xgb(get_param=False):
    global cpu_jobs
    params = {
        # all
        # 'learning_rate': 1e-1,
        # 'n_estimators': 80,
        # 'max_depth': 8,
        # 'min_child_weight': 3,
        # 'gamma': .2,
        # 'subsample': .8,
        # 'colsample_bytree': .8,

        # 10
        'learning_rate': 1e-2,
        'n_estimators': 1260,
        'max_depth': 8,
        'min_child_weight': 4,
        'gamma': .2,
        'subsample': .6,
        'colsample_bytree': .8,
        'scale_pos_weight': 1,
        'reg_alpha': 0,

        # 'learning_rate': 1e-1,
        # 'n_estimators': 80,
        # 'max_depth': 8,
        # 'min_child_weight': 3,
        # 'gamma': .2,
        # 'subsample': .8,
        # 'colsample_bytree': .8,
        # 'scale_pos_weight': 1,
        # 'reg_alpha': 0,

        'n_jobs': cpu_jobs,
        'seed': 0
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_child_weight': {'step': 1, 'min': 1, 'max': 'inf'},
        'gamma': {'step': .1, 'min': 0, 'max': 1},
        'subsample': {'step': .1, 'min': .1, 'max': 1},
        'colsample_bytree': {'step': .1, 'min': .1, 'max': 1},
        'scale_pos_weight': {'step': 1, 'min': 1, 'max': 10},
        'reg_alpha': {'step': .1, 'min': 0, 'max': 1},
    }

    grid_search_auto(steps, params, XGBClassifier())


def grid_search_lgb(get_param=False):
    params = {
        # 10
        'learning_rate': 1e-2,
        'n_estimators': 1200,
        'num_leaves': 51,
        'min_split_gain': 0,
        'min_child_weight': 1e-3,
        'min_child_samples': 22,
        'subsample': .8,
        'colsample_bytree': .8,

        # 'learning_rate': .1,
        # 'n_estimators': 90,
        # 'num_leaves': 50,
        # 'min_split_gain': 0,
        # 'min_child_weight': 1e-3,
        # 'min_child_samples': 21,
        # 'subsample': .8,
        # 'colsample_bytree': .8,

        'n_jobs': cpu_jobs,
        'random_state': 0
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 100, 'min': 1, 'max': 'inf'},
        'num_leaves': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_split_gain': {'step': .1, 'min': 0, 'max': 1},
        'min_child_weight': {'step': 1e-3, 'min': 1e-3, 'max': 'inf'},
        'min_child_samples': {'step': 1, 'min': 1, 'max': 'inf'},
        # 'subsample': {'step': .1, 'min': .1, 'max': 1},
        'colsample_bytree': {'step': .1, 'min': .1, 'max': 1},
    }

    grid_search_auto(steps, params, LGBMClassifier())


def grid_search_cat(get_param=False):
    params = {
        # 10
        'learning_rate': 1e-2,
        'n_estimators': 3600,
        'max_depth': 8,
        'max_bin': 127,
        'reg_lambda': 2,
        'subsample': .7,

        # 'learning_rate': 1e-1,
        # 'iterations': 460,
        # 'depth': 8,
        # 'l2_leaf_reg': 8,
        # 'border_count': 37,

        # 'ctr_border_count': 16,
        'one_hot_max_size': 2,
        'bootstrap_type': 'Bernoulli',
        'leaf_estimation_method': 'Newton',
        'random_state': 0,
        'verbose': False,
        'eval_metric': 'AUC',
        'thread_count': cpu_jobs
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 100, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'max_bin': {'step': 1, 'min': 1, 'max': 255},
        'reg_lambda': {'step': 1, 'min': 0, 'max': 'inf'},
        'subsample': {'step': .1, 'min': .1, 'max': 1},
        'one_hot_max_size': {'step': 1, 'min': 0, 'max': 255},
    }

    grid_search_auto(steps, params, CatBoostClassifier())


def grid_search_rf(criterion='gini', get_param=False):
    if criterion == 'gini':
        params = {
            # 10
            'n_estimators': 3090,
            'max_depth': 15,
            'min_samples_split': 2,
            'min_samples_leaf': 1,

            'criterion': 'gini',
            'random_state': 0
        }
    else:
        params = {
            'n_estimators': 3110,
            'max_depth': 13,
            'min_samples_split': 70,
            'min_samples_leaf': 10,
            'criterion': 'entropy',
            'random_state': 0
        }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 2, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 2, 'min': 1, 'max': 'inf'},
    }

    grid_search_auto(steps, params, RandomForestClassifier())


def grid_search_et(criterion='gini', get_param=False):
    if criterion == 'gini':
        params = {
            # 10
            'n_estimators': 3060,
            'max_depth': 22,
            'min_samples_split': 12,
            'min_samples_leaf': 1,

            'criterion': 'gini',
            'random_state': 0,
        }
    else:
        params = {
            'n_estimators': 3100,
            'max_depth': 13,
            'min_samples_split': 70,
            'min_samples_leaf': 10,
            'criterion': 'entropy',
            'random_state': 0
        }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 2, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 2, 'min': 1, 'max': 'inf'},
    }

    grid_search_auto(steps, params, ExtraTreesClassifier())


def train_gbdt(model=False):
    global log

    params = grid_search_gbdt(True)
    clf = GradientBoostingClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'gbdt'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', subsample: %.1f' % params['subsample']
    log += '\n\n'

    return train(clf)


def train_xgb(model=False):
    global log

    params = grid_search_xgb(get_param=True)

    clf = XGBClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'xgb'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_child_weight: %d' % params['min_child_weight']
    log += ', gamma: %.1f' % params['gamma']
    log += ', subsample: %.1f' % params['subsample']
    log += ', colsample_bytree: %.1f' % params['colsample_bytree']
    log += '\n\n'

    return train(clf)


def train_lgb(model=False):
    global log

    params = grid_search_lgb(True)

    clf = LGBMClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'lgb'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', num_leaves: %d' % params['num_leaves']
    log += ', min_split_gain: %.1f' % params['min_split_gain']
    log += ', min_child_weight: %.4f' % params['min_child_weight']
    log += ', min_child_samples: %d' % params['min_child_samples']
    log += ', subsample: %.1f' % params['subsample']
    log += ', colsample_bytree: %.1f' % params['colsample_bytree']
    log += '\n\n'

    return train(clf)


def train_cat(model=False):
    global log

    params = grid_search_cat(True)

    clf = CatBoostClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'cat'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', iterations: %d' % params['iterations']
    log += ', depth: %d' % params['depth']
    log += ', l2_leaf_reg: %d' % params['l2_leaf_reg']
    log += ', border_count: %d' % params['border_count']
    log += ', subsample: %d' % params['subsample']
    log += ', one_hot_max_size: %d' % params['one_hot_max_size']
    log += '\n\n'

    return train(clf)


def train_rf(clf):
    global log

    params = clf.get_params()
    log += 'rf'
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', criterion: %s' % params['criterion']
    log += '\n\n'

    return train(clf)


def train_rf_gini(model=False):
    clf = RandomForestClassifier().set_params(**grid_search_rf('gini', True))
    if model:
        return clf
    return train_rf(clf)


def train_rf_entropy():
    clf = RandomForestClassifier().set_params(**grid_search_rf('entropy', True))

    return train_rf(clf)


def train_et(clf):
    global log

    params = clf.get_params()
    log += 'et'
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', criterion: %s' % params['criterion']
    log += '\n\n'

    return train(clf)


def train_et_gini(model=False):
    clf = ExtraTreesClassifier().set_params(**grid_search_et('gini', True))
    if model:
        return clf
    return train_et(clf)


def train_et_entropy():
    clf = ExtraTreesClassifier().set_params(**{
        'n_estimators': 3100,
        'max_depth': 13,
        'min_samples_split': 70,
        'min_samples_leaf': 10,
        'criterion': 'entropy',
        'random_state': 0
    })

    return train_et(clf)


# train the model, clf = classifier分类器
def train(clf):
    global log

    data = feature.get_train_data()

    train_data, test_data = train_test_split(data,
                                             # train_size=1000,
                                             train_size=100000,
                                             random_state=0,
                                             # shuffle=True
                                             )

    _, test_data = train_test_split(data, random_state=0)

    X_train = train_data.copy().drop(columns='Coupon_id')
    y_train = X_train.pop('label')

    clf = fit_eval_metric(clf, X_train, y_train)

    X_test = test_data.copy().drop(columns='Coupon_id')
    y_test = X_test.pop('label')

    y_true, y_pred = y_test, clf.predict(X_test)
    # log += '%s\n' % classification_report(y_test, y_pred)
    log += '  accuracy: %f\n' % accuracy_score(y_true, y_pred)
    y_score = clf.predict_proba(X_test)[:, 1]
    log += '       auc: %f\n' % roc_auc_score(y_true, y_score)

    # coupon average auc 最终的评价指标对每个优惠券coupon_id单独计算预测结果的AUC值
    coupons = test_data.groupby('Coupon_id').size().reset_index(name='total')
    aucs = []
    for _, coupon in coupons.iterrows():
        if coupon.total > 1:
            X_test = test_data[test_data.Coupon_id == coupon.Coupon_id].copy()
            X_test.drop(columns='Coupon_id', inplace=True)

            # 需要去除那些只有一个标签的券，比如某张券全都是1或全都是0，这样的券无法计算AUC值
            # 相比于召回率、精确率、F1值，数据类不平衡时，AUC表现更好。
            if len(X_test.label.unique()) != 2:
                continue

            y_true = X_test.pop('label')
            y_score = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_true, y_score))

    log += 'coupon auc: %f\n\n' % np.mean(aucs)

    logger.set_logger(log)

    return clf

# def myauc(test):
#     testgroup = test.groupby(['coupon_id'])
#     aucs = []
#     for i in testgroup:
#         tmpdf = i[1]
#         if len(tmpdf['label'].unique()) != 2:
#             continue
#         fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred'], pos_label=1)
#         aucs.append(auc(fpr,tpr))
#     return np.average(aucs)


def predict(model):
    path = 'cache_%s_predict.csv' % os.path.basename(__file__)

    if os.path.exists(path):
        X = pd.read_csv(path, parse_dates=['Date_received'])
    else:
        offline, online = feature.get_preprocess_data()

        # 2016-03-16 ~ 2016-06-30，训练集日期到6-30为止且没有6.15-6.30的领券数据
        start = '2016-03-16' # 感觉这里应该用2-16，和训练集保持一致4.5月时长的跨度
        offline = offline[(offline.Coupon_id == 0) & (start <= offline.Date) | (start <= offline.Date_received)]
        online = online[(online.Coupon_id == 0) & (start <= online.Date) | (start <= online.Date_received)]

        X = feature.get_preprocess_data(True)
        X = feature.get_offline_features(X, offline)
        X = feature.get_online_features(online, X)
        X.drop_duplicates(inplace=True)
        X.fillna(0, inplace=True)
        X.to_csv(path, index=False)

    sample_submission = X[['User_id', 'Coupon_id', 'Date_received']].copy()
    sample_submission.Date_received = sample_submission.Date_received.dt.strftime('%Y%m%d')
    feature.drop_columns(X, True)

    if model is 'blending':
        predict = blending(X)
    else:
        clf = eval('train_%s' % model)()
        print(clf)
        predict = clf.predict_proba(X)[:, 1]

    sample_submission['Probability'] = predict
    sample_submission.to_csv('submission_%s.csv' % model,
                             #  float_format='%.5f',
                             index=False, header=False)

# Blending：用不相交的数据训练不同的 Base Model，将它们的输出取（加权）平均。实现简单，但对训练数据利用少了。
# 这里的实现其实是stacking的思想
def blending(predict_X=None):
    global log
    log += '\n'

    X = feature.get_train_data().drop(columns='Coupon_id')
    # X = X[:2000]
    y = X.pop('label')

    X = np.asarray(X)
    y = np.asarray(y)

    _, X_submission, _, y_test_blend = train_test_split(X, y,
                                                        random_state=0
                                                        )

    if predict_X is not None:
        X_submission = np.asarray(predict_X)

    X, _, y, _ = train_test_split(X, y,
                                  train_size=100000,
                                  # train_size=1000,
                                  random_state=0
                                  )

    # np.random.seed(0)
    # idx = np.random.permutation(y.size)
    # X = X[idx]
    # y = y[idx]

    skf = StratifiedKFold()
    clfs = ['gbdt', 'xgb', 'lgb', 'cat',
            # 'rf_gini', 'et_gini'
            ]

    blend_X_train = np.zeros((X.shape[0], len(clfs)))
    blend_X_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, v in enumerate(clfs):
        # 拿到设置好参数的model，不训练
        clf = eval('train_%s' % v)(True)

        aucs = []
        dataset_blend_test_j = []

        # 默认n_splits=3, 每次用部分数据训练模型，再对剩余数据做出预测，一共3次，正好对所有的数据都进行了预测
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = fit_eval_metric(clf, X_train, y_train)

            y_submission = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, y_submission))
            # 记录基模型test data的结果
            blend_X_train[test_index, j] = y_submission

            # 得到基模型对最终预测数据的预测结果
            dataset_blend_test_j.append(clf.predict_proba(X_submission)[:, 1])
            # print(len(dataset_blend_test_j), len(dataset_blend_test_j[0])) # 2D array 1 * 500 -> 2 * 500 -> n * 500

        log += '%7s' % v + ' auc: %f\n' % np.mean(aucs)

        # 对该基模型的预测取均值
        # 这里要按行算均值， 因为array的形式是[[500],[500],[500]...], 每次append 的都是一个[500]
        blend_X_test[:, j] = np.asarray(dataset_blend_test_j).mean(0) # .T.mean(1) = .mean(0)

    #多个基模型融合
    print('blending')
    clf = LogisticRegression()
    # clf = GradientBoostingClassifier()
    clf.fit(blend_X_train, y)
    # predict_proba 预测属于某标签的概率
    y_submission = clf.predict_proba(blend_X_test)[:, 1]
    # print("y_submission:", y_submission)

    # Linear stretch of predictions to [0,1]
    # 最终的评价指标对每个优惠券coupon_id单独计算预测结果的AUC值，再对所有优惠券的AUC值求平均作为最终的评价标准。
    # 明白了AUC的计算方式，便知道了重要的不是预测概率的绝对值，而是其相对值。
    # temp.pred = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(temp['pred'].values.reshape(-1, 1))
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    print("y_submission:", y_submission)

    if predict_X is not None:
        return y_submission
    log += '\n  blend auc: %f\n\n' % roc_auc_score(y_test_blend, y_submission)


if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    log = '%s\n' % start.strftime('%Y-%m-%d %H:%M:%S')
    cpu_jobs = os.cpu_count() - 1
    date_null = pd.to_datetime('1970-01-01', format='%Y-%m-%d')

    # feature_importance_score()

    grid_search_gbdt()
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

    log += 'time: %s\n' % str((datetime.datetime.now() - start)).split('.')[0]
    log += '----------------------------------------------------\n'
    # open('%s.log' % os.path.basename(__file__), 'a').write(log)
    print(log)

'''
未归一化前的y_submission：
y_submission: [0.04383296 0.03990142 0.07997316 0.04464753 0.04240286 0.04922496
 0.04263823 0.03990922 0.07624336 0.04091093 0.0397906  0.04065677
 0.04114965 0.04093842 0.04156598 0.03976165 0.03971192 0.04222231
 0.14613494 0.0421384  0.04275014 0.04138168 0.03976751 0.04230845
 0.04097935 0.04846415 0.04183242 0.04248478 0.03973803 0.07768046
 0.04067799 0.04235428 0.04063239 0.04081772 0.06108267 0.04571562]
'''