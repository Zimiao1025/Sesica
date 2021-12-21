import joblib
import numpy as np
from sklearn.svm import SVC

from utils.util_ctrl import file_ctrl
from utils.util_eval import evaluation


def svm_train(train_x, train_y, val_x, val_y, val_g, model_path, params):
    # get the cost and gamma
    c_range = params['cost']
    g_range = params['gamma']
    # save the dictionary of parameters and their corresponding evaluation indicators
    metric_dict = {}
    for c in c_range:
        for g in g_range:
            print('** cost: %d  |  gamma: %d **' % (c, g))
            clf = SVC(C=2 ** c, gamma=2 ** g, probability=True)
            # print(train_x.shape)
            clf.fit(train_x, train_y)
            val_prob = clf.predict_proba(val_x)[:, 1]
            # metric: auc, aupr, ndcg@k, roc@k
            prefix = 'svm_c_' + str(c) + '_g_' + str(g) + '_'
            abs_prefix = model_path + prefix
            np.save(abs_prefix+'valid_prob.npy', val_prob)
            metric_list = evaluation(params['metrics'], val_y, val_prob, val_g, abs_prefix+'valid_eval.csv')
            print(' Train '.center(36, '*'))
            print('Evaluation on validation dataset: ', metric_list[0])
            print('\n')
            metric_dict[(c, g)] = metric_list  # For example, params['metric'] = acc
    # sort from large to small
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)
    # select top_n best model
    top_n = params['top_n']['svm']
    # the balanced benchmark dataset for fitting
    prefix_list = []
    for i in range(top_n):
        hp = results_order[i][0]
        clf = SVC(C=2 ** hp[0], gamma=2 ** hp[1], probability=True)
        clf.fit(train_x, train_y)
        prefix = 'svm_c_' + str(hp[0]) + '_g_' + str(hp[1]) + '_'
        prefix_list.append(prefix)
        joblib.dump(clf, model_path+prefix+'model[top_' + str(i+1) + '].pkl')

    file_ctrl(prefix_list, model_path)
