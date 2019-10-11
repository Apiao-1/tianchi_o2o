import datetime
import os

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from xgboost.sklearn import XGBClassifier
from tianchi_o2o import model
from tianchi_o2o import feature
from tianchi_o2o import logger

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

    # feature.analysis()
    # feature.get_train_data()
    # feature.detect_duplicate_columns()
    # feature.feature_importance_score()

    # model.grid_search_gbdt()

    # model.train_gbdt()
    # model.predict('gbdt')

    # model.grid_search_xgb()
    # train_xgb()
    # model.predict('xgb')

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

    model.blending()
    # model.predict('blending')

    log = logger.get_logger()
    log += 'time: %s\n' % str((datetime.datetime.now() - start)).split('.')[0]
    log += '----------------------------------------------------\n'
    open('%s.log' % os.path.basename(__file__), 'a').write(log)

