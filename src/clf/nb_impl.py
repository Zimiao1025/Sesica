import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from utils.util_ctrl import file_ctrl
from utils.util_eval import evaluation


def nb_train(nb_model, train_x, train_y, val_x, val_y, val_g, model_path, params):
    # For MultinomialNB and BernoulliNB, [0.0, 1.0]
    alpha_range = params['nb_alpha'] if nb_model != 'gnb' else [1.0]
    # save the dictionary of parameters and their corresponding evaluation indicators
    metric_dict = {}
    for alpha in alpha_range:
        if nb_model == 'gnb':
            clf = GaussianNB()
        elif nb_model == 'mnb':
            clf = MultinomialNB(alpha=alpha)
        else:
            clf = BernoulliNB(alpha=alpha)

        clf.fit(train_x, train_y)
        val_prob = clf.predict_proba(val_x)[:, 1]
        # metric: auc, aupr, ndcg@k, roc@k
        prefix = nb_model + '_' + str(alpha) + '_'
        abs_prefix = model_path + prefix
        np.save(abs_prefix + 'valid_prob.npy', val_prob)
        metric_list = evaluation(params['metrics'], val_y, val_prob, val_g, abs_prefix + 'valid_eval.csv')
        print(' Train '.center(36, '*'))
        print('Evaluation on validation dataset: ', metric_list[0])
        print('\n')
        metric_dict[alpha] = metric_list  # For example, params['metric'] = acc
        # sort from large to small
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)

    # select top_n best model
    top_n = params['top_n'][nb_model]
    # the balanced benchmark dataset for fitting
    prefix_list = []
    for i in range(top_n):
        hp = results_order[i][0]
        if nb_model == 'gnb':
            clf = GaussianNB()
        elif nb_model == 'mnb':
            clf = MultinomialNB(alpha=hp)
        else:
            clf = BernoulliNB(alpha=hp)
        clf.fit(train_x, train_y)
        prefix = nb_model + '_' + str(hp) + '_'
        prefix_list.append(prefix)
        joblib.dump(clf, model_path+prefix+'model[top_' + str(i+1) + '].pkl')

    file_ctrl(prefix_list, model_path)
