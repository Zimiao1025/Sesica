from sklearn.neural_network import MLPClassifier

from utils.util_eval import performance


def mlp_train(train_x, train_y, val_x, val_y, params):
    # 获取激活函数和网络结构的范围
    act_list = params['act']  # [logistic, tanh, relu]
    hls_list = params['hls']  # [(256, 128), (512, 128), (256, 64)]
    # 保存参数及其对应评价指标的字典
    metric_dict = {}
    val_prob_dict = {}
    for act in act_list:
        for hls in hls_list:
            # solver default=adam
            clf = MLPClassifier(activation=act, hidden_layer_sizes=hls, probability=True)
            clf.fit(train_x, train_y)
            val_y_hat = clf.predict(val_x)
            val_prob = clf.predict_proba(val_x)[:, 1]
            metric = performance(val_y, val_y_hat, val_prob)
            metric_dict[(act, hls)] = metric[params['metric']]  # For example, params['metric'] = acc
            val_prob_dict[(act, hls)] = val_prob
    # 根据val对字典由大到小进行排序, 得到:  [((0, 1), 0.962), ((4, 3), 2), ...]
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)
    # 根据选择的的最好的模型个数，输出对应的预测概率和超参数
    top_n = params['top_n'][4]  # params['top_n'] = [3, 4, ..., 1] 每个数字对应一个算法 |搭建网站时需要注意
    opt_hps = []  # optimal hyper parameters
    scores = []
    for i in range(top_n):
        hp = results_order[i][0]
        opt_hps.append(hp)
        scores.append(val_prob_dict[hp])

    # 扩展部分 绘制top_n的ROC曲线等
    return opt_hps, scores


def mlp_predict(train_x, train_y, test_x, test_y, opt_hps, params):
    metric_list = []
    test_prob_list = []
    for act, hls in opt_hps:
        clf = MLPClassifier(activation=act, hidden_layer_sizes=hls, probability=True)
        clf.fit(train_x, train_y)
        test_y_hat = clf.predict(test_x)
        test_prob = clf.predict_proba(test_x)[:, 1]
        metric = performance(test_y, test_y_hat, test_prob)
        metric_list.append(metric[params['metric']])  # For example, params['metric'] = acc
        test_prob_list.append(test_prob)

    return metric_list, test_prob_list
