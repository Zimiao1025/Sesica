import joblib
import pandas as pd
from sklearn.svm import SVC

from utils.util_eval import evaluation, save_params


def svm_train(train_x, train_y, val_x, val_y, val_g, model_path, params):
    # get the cost and gamma
    c_range = params['svm_c']
    g_range = params['svm_g']
    metric_dict = {}
    for c in c_range:
        for g in g_range:
            print('** C: %d  |  gamma: %d **' % (c, g))
            clf = SVC(C=2 ** c, gamma=2 ** g, probability=True)
            clf.fit(train_x, train_y)
            val_prob = clf.predict_proba(val_x)[:, 1]
            # metric: auc, aupr, ndcg@k, roc@k
            metric_df = evaluation(params['metrics'], val_y, val_prob, val_g)
            metric_list = metric_df.mean().tolist()
            print('Evaluation on validation dataset: %s = %.4f\n' % (params['metrics'][0], metric_list[0]))
            metric_dict[(c, g)] = metric_list
    # sort from large to small
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)
    # select top_n best model
    top_n = params['top_n']['svm']
    # save the best params to csv file
    save_params(top_n, results_order, params['clf_param_keys']['svm'], model_path)
    prob_dict = {}
    for i in range(top_n):
        hp = results_order[i][0]
        clf = SVC(C=2 ** hp[0], gamma=2 ** hp[1], probability=True)
        clf.fit(train_x, train_y)
        joblib.dump(clf, model_path + 'model[top_' + str(i+1) + '].pkl')
        val_prob = clf.predict_proba(val_x)[:, 1]
        prob_dict['top_' + str(i+1)] = val_prob
        # metric: auc, aupr, ndcg@k, roc@k
        metric_df = evaluation(params['metrics'], val_y, val_prob, val_g)
        metric_df.to_csv(model_path + 'top_' + str(i+1) + '_eval_results.csv')
        metric_list = metric_df.mean().tolist()
        print('Evaluation result of top_%d model: %s = %.4f\n' % (i+1, params['metrics'][0], metric_list[0]))

    prob_df = pd.DataFrame(prob_dict)
    prob_df.to_csv(model_path + 'valid_prob.csv')
