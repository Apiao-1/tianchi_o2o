# 封装的Stacking和Bagging
# 本模型在银杏大数据竞赛中拿了第一名
# 在jdata比赛baseline中开源

'''
https://tianchi.aliyun.com/notebook-ai/detail?postId=22709
# 使用示例：模型训练和预测
model2 = SBBTree(params=params, stacking_num=1, bagging_num=1,  bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=200)
model2.fit(X,y)
y_pred_out = model2.predict(X_pred)
'''

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


class SBBTree():
    """Stacking,Bootstap,Bagging----SBBTree"""
    """ author：Cookly 洪鹏飞 """

    def __init__(self, params, stacking_num, bagging_num, bagging_test_size, num_boost_round, early_stopping_rounds):
        """
        Initializes the SBBTree.
        Args:
          params : lgb params.
          stacking_num : k_flod stacking.
          bagging_num : bootstrap num.
          bagging_test_size : bootstrap sample rate.
          num_boost_round : boost num.
          early_stopping_rounds : early_stopping_rounds.
        """
        self.params = params
        self.stacking_num = stacking_num
        self.bagging_num = bagging_num
        self.bagging_test_size = bagging_test_size
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.model = lgb
        self.stacking_model = []
        self.bagging_model = []

    def fit(self, X, y):
        """ fit model."""
        if self.stacking_num > 1:
            layer_train = np.zeros((X.shape[0], 2))
            self.SK = StratifiedKFold(n_splits=self.stacking_num, shuffle=True, random_state=1)
            for k, (train_index, test_index) in enumerate(self.SK.split(X, y)):
                X_train = X[train_index]
                y_train = y[train_index]
                X_test = X[test_index]
                y_test = y[test_index]

                lgb_train = lgb.Dataset(X_train, y_train)
                lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

                gbm = lgb.train(self.params,
                                lgb_train,
                                num_boost_round=self.num_boost_round,
                                valid_sets=lgb_eval,
                                early_stopping_rounds=self.early_stopping_rounds)

                self.stacking_model.append(gbm)

                pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
                layer_train[test_index, 1] = pred_y

            # 相当于在X的后面增加一列y，先通过Stacking得到预测的y，然后将其加入x，作为一个新的特征
            X = np.hstack((X, layer_train[:, 1].reshape((-1, 1))))
        else:
            pass
        # 进一步用bagging和新的X再训练多个lgb的模型
        for bn in range(self.bagging_num):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.bagging_test_size, random_state=bn)

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

            gbm = lgb.train(self.params,
                            lgb_train,
                            num_boost_round=10000,
                            valid_sets=lgb_eval,
                            early_stopping_rounds=200)

            self.bagging_model.append(gbm)

    def predict(self, X_pred):
        """ predict test data. """
        if self.stacking_num > 1:
            test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
            for sn, gbm in enumerate(self.stacking_model):
                pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
                test_pred[:, sn] = pred
            # 把第一层预测的y当做新的特征
            X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1, 1))))
        else:
            pass
        # 在第二层结合新的特征再产生结果
        for bn, gbm in enumerate(self.bagging_model):
            pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
            if bn == 0:
                pred_out = pred
            else:
                pred_out += pred
        return pred_out / self.bagging_num