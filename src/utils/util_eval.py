import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc, precision_recall_curve


def roc_score(y_true, y_prob, level):
    y_true_sorted = y_true[np.argsort(-y_prob)]
    # 统计所有的true positive
    tp = np.count_nonzero(y_true_sorted)
    # print(tp)
    # 统计level个false positive之前的tp之和
    all_fp = np.count_nonzero(~y_true_sorted.astype(bool))
    if all_fp < level:
        # 如果不足就填充False，因为，没检索出来
        # fp = all_fp
        yt_level = y_true_sorted
    else:
        df = pd.DataFrame(y_true_sorted.astype(bool), columns=["label"])
        fp_index = df.loc[~df.loc[:, "label"]].index[level - 1]
        yt_level = df.loc[:fp_index].to_numpy(int).reshape((-1,))
    fp = np.count_nonzero(~yt_level.astype(bool))
    # print(fp)

    # print(yt_level)
    cumsum_yt_level = np.cumsum(yt_level)
    area = cumsum_yt_level[~yt_level.astype(bool)].sum()
    # print(area)
    if tp != 0 and fp != 0:
        roc_at_score = area / (tp * fp)
    elif tp != 0 and fp == 0:
        roc_at_score = 1.0
    elif tp == 0 and fp != 0:
        roc_at_score = 0.0
    else:
        raise Exception("unexpected error")
    return roc_at_score


def dcg_score_k(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def dcg_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score_k(y_true, y_score, k):
    dcg = dcg_score_k(y_true, y_score, k)
    idcg = dcg_score_k(y_true, y_true, k)
    try:
        ndcg = dcg / idcg
    except ZeroDivisionError:
        ndcg = 0.0

    return ndcg


def ndcg_score(y_true, y_score):
    dcg = dcg_score(y_true, y_score)
    idcg = dcg_score(y_true, y_true)
    try:
        ndcg = dcg / idcg
    except ZeroDivisionError:
        ndcg = 0.0

    return ndcg


def auc_score(true_labels, pred_prob):
    fpr_ind, tpr_ind, thresholds_ind = roc_curve(true_labels, pred_prob)
    try:
        auc_val = auc(fpr_ind, tpr_ind)
    except ZeroDivisionError:
        auc_val = 0.0

    return auc_val


def aupr_score(true_labels, pred_prob):
    precision, recall, _ = precision_recall_curve(true_labels, pred_prob)
    try:
        aupr_val = auc(recall, precision)
    except ZeroDivisionError:
        aupr_val = 0.0
    return aupr_val


def group_eval(metric, y_true, y_prob):
    # aupr, auc, ndcg@10, roc@1, roc@50
    if metric == 'aupr':
        eval_score = aupr_score(y_true, y_prob)
    elif metric == 'auc':
        eval_score = auc_score(y_true, y_prob)
    elif metric == 'ndcg':
        eval_score = ndcg_score(y_true, y_prob)
    elif 'ndcg' in metric and metric != 'ndcg':
        k = int(metric.split('@')[1])
        eval_score = ndcg_score_k(y_true, y_prob, k)
    else:
        k = int(metric.split('@')[1])
        eval_score = roc_score(y_true, y_prob, k)
    return eval_score


def evaluation(metrics, val_y, val_prob, val_g):
    # val_y, val_prob, val_g --> array, array, array
    count = 0
    # Evaluate by group and then calculate the mean.
    eval_res = []
    for group in val_g:
        y_true = val_y[count: count + group]
        y_prob = val_prob[count: count + group]

        group_eval_dict = {}
        for metric in metrics:
            group_eval_dict[metric] = group_eval(metric, y_true, y_prob)
        eval_res.append(group_eval_dict)
        count += group

    df = pd.DataFrame(eval_res)
    return df


def save_params(top_n, results, param_keys, model_path):
    param_list = []
    for i in range(top_n):
        tmp_dict = {'top_n': i + 1}
        for param_key, param_val in zip(param_keys, results[i][0]):
            tmp_dict[param_key] = param_val
        param_list.append(tmp_dict)
    params_df = pd.DataFrame(param_list)
    params_df.to_csv(model_path + 'params.csv')
