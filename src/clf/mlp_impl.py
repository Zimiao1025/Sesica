import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from utils.util_ctrl import file_ctrl
from utils.util_eval import evaluation


def mlp_train(train_x, train_y, val_x, val_y, val_g, model_path, params):
    # get act func and hidden layer size
    act_list = params['act']  # [logistic, tanh, relu]
    hls_list = params['hls']  # [(256, 128), (512, 128), (256, 64)]
    # save the dictionary of parameters and their corresponding evaluation indicators
    metric_dict = {}
    for act in act_list:
        for hls in hls_list:
            # solver default=adam
            clf = MLPClassifier(activation=act, hidden_layer_sizes=hls)
            clf.fit(train_x, train_y)
            val_prob = clf.predict_proba(val_x)[:, 1]
            # metric: auc, aupr, ndcg@k, roc@k
            prefix = 'mlp_act_' + act + '_hls_' + '_'.join(list(map(str, list(hls)))) + '_'
            abs_prefix = model_path + prefix
            np.save(abs_prefix + 'valid_prob.npy', val_prob)
            metric_list = evaluation(params['metrics'], val_y, val_prob, val_g, abs_prefix + 'valid_eval.csv')
            print(' Train '.center(36, '*'))
            print('Evaluation on validation dataset: ', metric_list[0])
            print('\n')
            metric_dict[(act, hls)] = metric_list  # For example, params['metric'] = acc
    # sort from large to small
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)

    # select top_n best model
    top_n = params['top_n']['mlp']
    # the balanced benchmark dataset for fitting
    prefix_list = []
    for i in range(top_n):
        hp = results_order[i][0]
        clf = MLPClassifier(activation=hp[0], hidden_layer_sizes=hp[1])
        clf.fit(train_x, train_y)
        prefix = 'mlp_act_' + hp[0] + '_hls_' + '_'.join(list(map(str, list(hp[1])))) + '_'
        prefix_list.append(prefix)
        joblib.dump(clf, model_path+prefix+'model[top_' + str(i+1) + '].pkl')

    file_ctrl(prefix_list, model_path)
