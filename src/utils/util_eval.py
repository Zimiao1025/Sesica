import math
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


def performance(origin_labels, predict_labels, deci_value):
    """evaluations used to evaluate the performance of the model.
    :param deci_value: decision values used for ROC and AUC.
    :param origin_labels: true values of the data set.
    :param predict_labels: predicted values of the data set.
    """
    if len(origin_labels) != len(predict_labels):
        raise ValueError("The number of the original labels must equal to that of the predicted labels.")
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    metric = {}
    for i in range(len(origin_labels)):
        if origin_labels[i] == 1 and predict_labels[i] == 1:
            tp += 1.0
        elif origin_labels[i] == 1 and predict_labels[i] == 0:
            fn += 1.0
        elif origin_labels[i] == 0 and predict_labels[i] == 1:
            fp += 1.0
        elif origin_labels[i] == 0 and predict_labels[i] == 0:
            tn += 1.0

    try:
        sn = tp / (tp + fn)
    except ZeroDivisionError:
        sn = 0.0
    try:
        sp = tn / (fp + tn)
    except ZeroDivisionError:
        sp = 0.0
    try:
        acc = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        acc = 0.0
    try:
        mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    except ZeroDivisionError:
        mcc = 0.0
    try:
        auc = roc_auc_score(origin_labels, deci_value)
    except ValueError:  # modify in 2020/9/13
        auc = 0.0
    b_acc = (sn + sp) / 2
    # 写入字典
    metric['acc'] = acc
    metric['mcc'] = mcc
    metric['auc'] = auc
    metric['b_acc'] = b_acc
    metric['sn'] = sn
    metric['sp'] = sp
    return metric


def cal_roc_mu_at(level, y_true):
    # 统计所有的true positive
    tp = np.count_nonzero(y_true)
    # print(tp)
    # 统计level个false positive之前的tp之和
    all_fp = np.count_nonzero(~y_true.astype(bool))
    if all_fp < level:
        # 如果不足就填充False，因为，没检索出来
        # fp = all_fp
        yt_level = y_true
    else:
        df = pd.DataFrame(y_true.astype(bool), columns=["label"])
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
