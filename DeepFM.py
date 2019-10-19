import deepctr
from deepctr import inputs

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from tianchi_o2o import feature
from tqdm import tqdm
from tianchi_o2o import logger

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 200)

if __name__ == '__main__':
    # global log
    # print(start.strftime('%Y-%m-%d %H:%M:%S'))
    # logger.init_logger()
    log = logger.get_logger()


    data = feature.get_train_data()

    sparse_features = ['weekday', 'day', 'discount_type']
    dense_features = [fea for fea in data.columns if fea not in sparse_features and fea not in ['Coupon_id','label']]

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_list = [inputs.SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]
    dense_feature_list = [inputs.DenseFeat(feat, 1, ) for feat in dense_features]

    dnn_feature_columns = linear_feature_columns = sparse_feature_list + dense_feature_list

    train_data, test_data = train_test_split(data,
                                             # train_size=1000,
                                             train_size=100000,
                                             random_state=0,
                                             # shuffle=True
                                             )

    _, test_data = train_test_split(data, random_state=0)

    X_train = train_data.copy().drop(columns='Coupon_id')
    y_train = X_train.pop('label')

    X_test = test_data.copy().drop(columns='Coupon_id')
    y_test = X_test.pop('label')

    feature_names = deepctr.inputs.get_feature_names(linear_feature_columns + dnn_feature_columns)

    X_train = {name:X_train[name] for name in feature_names}
    X_test = {name:X_test[name] for name in feature_names}

    checkpoint_predictions = []
    weights = []

    ### 模型训练并做一个简单的Ensembe

    for model_idx in range(2):
        print('【', 'model_{}'.format(model_idx + 1), '】')
        model = deepctr.models.DeepFM(linear_feature_columns, dnn_feature_columns,
            # dnn_hidden_units=(64, 64),
            dnn_use_bn=True,
            task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        for global_epoch in range(2):
            print('【', 'global_epoch_{}'.format(global_epoch + 1), '】')
            model.fit(
                X_train,
                y_train,
                batch_size=64,
                epochs=1,
                verbose=1)
            checkpoint_predictions.append(model.predict(X_test, batch_size=64).flatten())
            weights.append(2 ** global_epoch)
    # clf = fit_eval_metric(clf, X_train, y_train)

    y_true, y_pred = y_test, list(np.average(checkpoint_predictions, weights=weights, axis=0))
    test_data['y_pred'] = y_pred
    print(test_data.head())
    # log += '%s\n' % classification_report(y_test, y_pred)
    # log += '  accuracy: %f\n' % accuracy_score(y_true, y_pred)
    # y_score = clf.predict_proba(X_test)[:, 1]
    log += '       auc: %f\n' % roc_auc_score(y_true, y_pred)

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

            # y_true = X_test.pop('label')
            # print(y_true)
            # y_score = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(X_test.label, X_test.y_pred))

    log += 'coupon auc: %f\n\n' % np.mean(aucs)

    logger.set_logger(log)

    print(log)