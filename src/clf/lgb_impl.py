import joblib
import pandas as pd
from lightgbm import LGBMClassifier as lgb

from utils.util_eval import evaluation, save_params


def lgb_train(bt_type, train_x, train_y, val_x, val_y, val_g, model_path, params):
    # get num_leaves and tree
    n_range = params[bt_type+'_n']
    t_range = params[bt_type+'_t']
    metric_dict = {}
    for n in n_range:
        for t in t_range:
            print('** num_leaves: %d  |  n_estimators: %d **' % (n, t))
            gbm = lgb(boosting_type=bt_type, objective='binary', random_state=1025, n_estimators=t, num_leaves=n)
            gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
            val_prob = gbm.predict_proba(val_x)[:, 1]
            # metric: auc, aupr, ndcg@k, roc@k
            # metric: auc, aupr, ndcg@k, roc@k
            metric_df = evaluation(params['metrics'], val_y, val_prob, val_g)
            metric_list = metric_df.mean().tolist()
            print('Evaluation on validation dataset: %s = %.4f\n' % (params['metrics'][0], metric_list[0]))
            metric_dict[(n, t)] = metric_list
    # sort from large to small
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)
    # select top_n best model
    top_n = params['top_n'][bt_type]
    # save the best params to csv file
    save_params(top_n, results_order, params['param_keys'][bt_type], model_path)
    prob_dict = {}
    for i in range(top_n):
        hp = results_order[i][0]
        gbm = lgb(boosting_type=bt_type, objective='binary', random_state=1025, n_estimators=hp[0], num_leaves=hp[1])
        gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)])
        joblib.dump(gbm, model_path + 'model[top_' + str(i + 1) + '].pkl')
        val_prob = gbm.predict_proba(val_x)[:, 1]
        prob_dict['top_' + str(i+1)] = val_prob
        # metric: auc, aupr, ndcg@k, roc@k
        metric_df = evaluation(params['metrics'], val_y, val_prob, val_g)
        metric_df.to_csv(model_path + 'top_' + str(i + 1) + '_eval_results.csv')
        metric_list = metric_df.mean().tolist()
        print('Evaluation result of top_%d model: %s = %.4f\n' % (i + 1, params['metrics'][0], metric_list[0]))

    prob_df = pd.DataFrame(prob_dict)
    prob_df.to_csv(model_path + 'valid_prob.csv')
