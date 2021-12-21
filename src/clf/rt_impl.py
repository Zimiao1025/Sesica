import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from utils.util_ctrl import file_ctrl
from utils.util_eval import evaluation


def rt_train(tree_model, train_x, train_y, val_x, val_y, val_g, model_path, params):
    # get n_estimators
    t_range = params['n_estimators']
    # save the dictionary of parameters and their corresponding evaluation indicators
    metric_dict = {}
    for t in t_range:
        clf = RandomForestClassifier(n_estimators=t, random_state=1025) if tree_model == 'rf' \
            else ExtraTreesClassifier(n_estimators=t, random_state=1025)
        clf.fit(train_x, train_y)
        val_prob = clf.predict_proba(val_x)[:, 1]
        # metric: auc, aupr, ndcg@k, roc@k
        prefix = tree_model + '_' + str(t) + '_'
        abs_prefix = model_path + prefix
        np.save(abs_prefix + 'valid_prob.npy', val_prob)
        metric_list = evaluation(params['metrics'], val_y, val_prob, val_g, abs_prefix + 'valid_eval.csv')
        print(' Train '.center(36, '*'))
        print('Evaluation on validation dataset: ', metric_list[0])
        print('\n')
        metric_dict[t] = metric_list  # For example, params['metric'] = acc
    # sort from large to small
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)

    # select top_n best model
    top_n = params['top_n'][tree_model]
    # the balanced benchmark dataset for fitting
    prefix_list = []
    for i in range(top_n):
        hp = results_order[i][0]
        clf = RandomForestClassifier(n_estimators=hp, random_state=1025) if tree_model == 'rf' \
            else ExtraTreesClassifier(n_estimators=hp, random_state=1025)
        clf.fit(train_x, train_y)
        prefix = tree_model + '_' + str(hp) + '_'
        prefix_list.append(prefix)
        joblib.dump(clf, model_path + prefix + 'model[top_' + str(i + 1) + '].pkl')

    file_ctrl(prefix_list, model_path)
