import joblib
import numpy as np
from lightgbm import LGBMClassifier as lgb
from sklearn.svm import SVC

from utils.util_ctrl import file_ctrl
from utils.util_eval import evaluation


def lgb_train(bt_type, train_x, train_y, val_x, val_y, val_g, model_path, params):
    # get num_leaves and tree
    n_range = params['num_leaves']
    t_range = params['n_estimators']
    # boosting_type = params['boosting_type']  # 'gbdt', 'dart', 'goss'
    # save the dictionary of parameters and their corresponding evaluation indicators
    metric_dict = {}
    for n in n_range:
        for t in t_range:
            gbm = lgb(boosting_type=bt_type, objective='binary', random_state=1025, n_estimators=t, num_leaves=n)
            gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
            val_prob = gbm.predict_proba(val_x)[:, 1]
            # metric: auc, aupr, ndcg@k, roc@k
            prefix = bt_type + '_n_' + str(n) + '_t_' + str(t) + '_'
            abs_prefix = model_path + prefix
            np.save(abs_prefix + 'valid_prob.npy', val_prob)
            metric_list = evaluation(params['metrics'], val_y, val_prob, val_g, abs_prefix + 'valid_eval.csv')
            print(' Train '.center(36, '*'))
            print('Evaluation on validation dataset: ', metric_list[0])
            print('\n')
            metric_dict[(n, t)] = metric_list  # For example, params['metric'] = acc
    # sort from large to small
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)

    # select top_n best model
    top_n = params['top_n'][bt_type]
    # the balanced benchmark dataset for fitting
    prefix_list = []
    for i in range(top_n):
        hp = results_order[i][0]
        gbm = lgb(boosting_type=bt_type, objective='binary', random_state=1025, n_estimators=hp[0], num_leaves=hp[1])
        gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
        prefix = bt_type + '_n_' + str(hp[0]) + '_t_' + str(hp[1]) + '_'
        prefix_list.append(prefix)
        joblib.dump(gbm, model_path+prefix+'model[top_' + str(i+1) + '].pkl')

    file_ctrl(prefix_list, model_path)
