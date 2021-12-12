import math

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
        r = sn
    except ZeroDivisionError:
        sn, r = 0.0, 0.0
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
    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        p = 0.0
    try:
        f1 = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1 = 0.0
    b_acc = (sn + sp) / 2
    # 写入字典
    metric['acc'] = acc
    metric['mcc'] = mcc
    metric['auc'] = auc
    metric['b_acc'] = b_acc
    metric['sn'] = sn
    metric['sp'] = sp
    metric['p'] = p
    metric['r'] = r
    metric['f1'] = f1
    return metric
