from lightgbm import LGBMClassifier as lgb

from utils.utils_eval import performance


def lgb_train(train_x, train_y, val_x, val_y, params):
    # 获取cost和gamma的范围
    n_range = params['num_leaves']
    t_range = params['lgb_tree']
    boosting_type = params['boosting_type']  # 'gbdt', 'dart', 'goss'
    # 保存参数及其对应评价指标的字典
    metric_dict = {}
    val_prob_dict = {}
    for n in n_range:
        for t in t_range:
            clf = lgb(boosting_type=boosting_type, objective='binary', n_estimators=t, num_leaves=n,
                      early_stopping_rounds=5, probability=True)
            clf.fit(train_x, train_y)
            val_y_hat = clf.predict(val_x)
            val_prob = clf.predict_proba(val_x)[:, 1]
            metric = performance(val_y, val_y_hat, val_prob)
            metric_dict[(n, t)] = metric[params['metric']]  # For example, params['metric'] = acc
            val_prob_dict[(n, t)] = val_prob
    # 根据val对字典由大到小进行排序, 得到:  [((0, 1), 0.962), ((4, 3), 2), ...]
    results_order = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)
    # 根据选择的的最好的模型个数，输出对应的预测概率和超参数
    top_n = params['top_n'][3]  # params['top_n'] = [3, 4, ..., 1] 每个数字对应一个算法 |搭建网站时需要注意
    opt_hps = []  # optimal hyper parameters
    scores = []
    for i in range(top_n):
        hp = results_order[i][0]
        opt_hps.append(hp)
        scores.append(val_prob_dict[hp])

    # 扩展部分 绘制top_n的ROC曲线等
    return opt_hps, scores


def lgb_predict(train_x, train_y, test_x, test_y, opt_hps, params):
    metric_list = []
    test_prob_list = []
    boosting_type = params['boosting_type']
    for n, t in opt_hps:
        clf = lgb(boosting_type=boosting_type, objective='binary', n_estimators=t, num_leaves=n,
                  early_stopping_rounds=5, probability=True)
        clf.fit(train_x, train_y)
        test_y_hat = clf.predict(test_x)
        test_prob = clf.predict_proba(test_x)[:, 1]
        metric = performance(test_y, test_y_hat, test_prob)
        print(' Test '.center(36, '*'))
        print(metric)
        print('\n')
        metric_list.append(metric[params['metric']])  # For example, params['metric'] = acc
        test_prob_list.append(test_prob)

    return metric_list, test_prob_list
