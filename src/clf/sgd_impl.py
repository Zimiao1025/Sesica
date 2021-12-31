import joblib
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.util_eval import evaluation, save_params


def sgd_train(train_x, train_y, val_x, val_y, val_g, model_path, params):
    # get max_iter
    m_range = params['sgd_m']
    # save the dictionary of parameters and their corresponding evaluation indicators
    metric_dict = {}
    for m in m_range:
        print('** max_iter: %d **' % m)
        clf = make_pipeline(StandardScaler(), SGDClassifier(loss='log', max_iter=m, learning_rate='constant',
                                                            eta0=0.001, random_state=1025))
        clf.fit(train_x, train_y)
        val_prob = clf.predict_proba(val_x)[:, 1]
        # metric: auc, aupr, ndcg@k, roc@k
        metric_df = evaluation(params['metrics'], val_y, val_prob, val_g)
        metric_list = metric_df.mean().tolist()
        print('Evaluation on validation dataset: %s = %.4f\n' % (params['metrics'][0], metric_list[0]))
        exit()
        metric_dict[(m,)] = metric_list
    # sort from large to small
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)
    # select top_n best model
    top_n = params['top_n']['sgd']
    # save the best params to csv file
    save_params(top_n, results_order, params['param_keys']['sgd'], model_path)
    prob_dict = {}
    for i in range(top_n):
        hp = results_order[i][0]
        clf = make_pipeline(StandardScaler(), SGDClassifier(loss='log', max_iter=hp[0], learning_rate='constant',
                                                            eta0=0.001, random_state=1025))
        clf.fit(train_x, train_y)
        joblib.dump(clf, model_path + 'model[top_' + str(i + 1) + '].pkl')
        val_prob = clf.predict_proba(val_x)[:, 1]
        prob_dict['top_' + str(i+1)] = val_prob
        # metric: auc, aupr, ndcg@k, roc@k
        metric_df = evaluation(params['metrics'], val_y, val_prob, val_g)
        metric_df.to_csv(model_path + 'top_' + str(i + 1) + '_eval_results.csv')
        metric_list = metric_df.mean().tolist()
        print('Evaluation result of top_%d model: %s = %.4f\n' % (i + 1, params['metrics'][0], metric_list[0]))

    prob_df = pd.DataFrame(prob_dict)
    prob_df.to_csv(model_path + 'valid_prob.csv')
