import joblib
import numpy as np
from lightgbm import LGBMRanker as lgb

from utils.util_eval import evaluation


def ltr_train(train_x, train_y, train_g, val_x, val_y, val_g, int_path, params):
    # get num_leaves and tree
    n_range = params['ltr_n']  # num_leaves
    t_range = params['ltr_t']  # n_estimators
    m_range = params['ltr_m']  # max_depth
    # boosting_type = params['boosting_type']  # 'gbdt', 'dart', 'goss'
    metric_dict = {}
    for m in m_range:
        for n in n_range:
            for t in t_range:
                print('** max_depth: %d  |  num_leaves: %d  |  n_estimators: %d **' % (m, n, t))
                gbm = lgb(boosting_type='dart', random_state=1025, max_depth=m, num_leaves=n, n_estimators=t)
                gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], group=train_g, eval_group=[val_g])
                val_prob = gbm.predict(val_x)
                # metric: auc, aupr, ndcg@k, roc@k
                metric_df = evaluation(params['metrics'], val_y, val_prob, val_g)
                metric_list = metric_df.mean().tolist()
                print('Evaluation on validation dataset for integrating: %s = %.4f\n' % (params['metrics'][0],
                                                                                         metric_list[0]))
                metric_dict[(m, n, t)] = metric_list
    # sort from large to small
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)
    # select best model
    hp = results_order[0][0]
    gbm = lgb(boosting_type='dart', random_state=1025, max_depth=hp[0], num_leaves=hp[1], n_estimators=hp[2])
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], group=train_g, eval_group=[val_g])
    joblib.dump(gbm, int_path + 'ltr_m_' + str(hp[0]) + '_n_' + str(hp[1]) + '_t_' + str(hp[2]) + '_model.pkl')
    val_prob = gbm.predict(val_x)
    np.save(int_path + 'prob.npy', val_prob)
    # metric: auc, aupr, ndcg@k, roc@k
    metric_df = evaluation(params['metrics'], val_y, val_prob, val_g)
    metric_df.to_csv(int_path + 'int_results.csv')
    metric_list = metric_df.mean().tolist()
    print('Final results for integration: %s = %.4f\n' % (params['metrics'][0], metric_list[0]))
