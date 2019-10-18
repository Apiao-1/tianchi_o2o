import deepctr
from deepctr import inputs
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve,roc_auc_score

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
    global log

    data = feature.get_train_data()[:2000]

    sparse_features = ['weekday', 'day', 'discount_type']
    dense_features = [fea for fea in data.columns if fea not in sparse_features and fea not in ['Coupon_id','label']]

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # print(X_train.describe())
    sparse_feature_list = [inputs.SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]
    dense_feature_list = [inputs.DenseFeat(feat, 0, ) for feat in dense_features]

    # fixlen_feature_columns = [inputs.SparseFeat(feat, data[feat].nunique())
    #                        for feat in sparse_features] + [inputs.DenseFeat(feat, 1,)
    #                       for feat in dense_features]
    # sparse_feature_list = dense_feature_list = fixlen_feature_columns

    # train_data, test_data = train_test_split(data,
    #                                          train_size=1000,
    #                                          # train_size=100000,
    #                                          random_state=0,
    #                                          # shuffle=True
    #                                          )
    #
    # _, test_data = train_test_split(data, random_state=0)
    data.drop(columns='Coupon_id')

    train, valid = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=0)
    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
                        [train[feat.name].values for feat in dense_feature_list]
    valid_model_input = [valid[feat.name].values for feat in sparse_feature_list] + \
                        [valid[feat.name].values for feat in dense_feature_list]

    checkpoint_predictions = []
    weights = []

    ### 模型训练并做一个简单的Ensembe

    for model_idx in range(2):
        print('【', 'model_{}'.format(model_idx + 1), '】')
        model = deepctr.models.DeepFM(
            sparse_feature_list,dense_feature_list,
            dnn_hidden_units=(64, 64),
            dnn_use_bn=True,
            task='binary')
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
        for global_epoch in range(2):
            print('【', 'global_epoch_{}'.format(global_epoch + 1), '】')
            model.fit(
                train_model_input,
                train['label'].values,
                batch_size=64,
                epochs=1,
                verbose=1)
            checkpoint_predictions.append(model.predict(valid_model_input, batch_size=64).flatten())
            weights.append(2 ** global_epoch)
    # clf = fit_eval_metric(clf, X_train, y_train)

    y_true, y_pred = valid['label'].values, np.average(checkpoint_predictions, weights=weights, axis=0)
    log += '       auc: %f\n' % roc_auc_score(y_true, y_pred)

    # avgAUC calculation
    predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
    valid1 = valid.copy()
    valid1['pred_prob'] = list(predictions)
    vg = valid1.groupby(['Coupon_id'])
    aucs = []
    for i in vg:
        tmpdf = i[1]
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    print('auc: ', np.average(aucs))

    log += 'coupon auc: %f\n\n' % np.mean(aucs)

    logger.set_logger(log)