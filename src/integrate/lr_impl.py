import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

from utils.util_eval import evaluation


def lr_train(train_x, train_y, val_x, val_y, val_g, int_path, params):
    # get the cost and gamma
    c_range = params['lr_c']
    metric_dict = {}
    for c in c_range:
        print('** lr_c: %d  **' % c)
        clf = LogisticRegression(C=2 ** c, random_state=1025)
        train_x, train_y = SMOTE(random_state=1025).fit_sample(train_x, train_y)
        clf.fit(train_x, train_y)
        val_prob = clf.predict_proba(val_x)[:, 1]
        # metric: auc, aupr, ndcg@k, roc@k
        metric_df = evaluation(params['metrics'], val_y, val_prob, val_g)
        metric_list = metric_df.mean().tolist()
        print('Evaluation on validation dataset for integrating: %s = %.4f\n' % (params['metrics'][0], metric_list[0]))
        metric_dict[(c,)] = metric_list
    # sort from large to small
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)
    # select best model
    hp = results_order[0][0]
    clf = LogisticRegression(C=2 ** hp[0], random_state=1025)
    train_x, train_y = SMOTE(random_state=1025).fit_sample(train_x, train_y)
    clf.fit(train_x, train_y)
    best_param = [{'lr_c': hp[0]}]
    pd.DataFrame(best_param).to_csv(int_path + 'params.csv')
    joblib.dump(clf, int_path + 'lr_model.pkl')
    val_prob = clf.predict_proba(val_x)[:, 1]
    np.save(int_path + 'prob.npy', val_prob)
    # metric: auc, aupr, ndcg@k, roc@k
    metric_df = evaluation(params['metrics'], val_y, val_prob, val_g)
    metric_df.to_csv(int_path + 'eval_results.csv')
    metric_list = metric_df.mean().tolist()
    print('Final results for integration: %s = %.4f\n' % (params['metrics'][0], metric_list[0]))
