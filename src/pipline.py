import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks

from clf.svm_impl import svm_train, svm_predict
from clf.rf_impl import rf_train, rf_predict
from clf.nb_impl import nb_train, nb_predict
from clf.lgb_impl import lgb_train, lgb_predict
from clf.mlp_impl import mlp_train, mlp_predict
from utils import utils_params


def load_dataset():
    # 加载训练数据
    train_data_pos = np.load('./output1/train_1.npy')
    train_data_neg = np.load('./output1/train_0.npy')
    # 将训练集负样本随机打乱
    np.random.shuffle(train_data_neg)
    # 对训练集负样本进行下采样，使正负样本比例为 1: 1
    sp_train_data_neg = train_data_neg[:len(train_data_pos)]
    train_data = np.vstack((train_data_pos, sp_train_data_neg))
    train_label = np.array([1.0] * len(train_data_pos) + [0.0] * len(sp_train_data_neg))

    # 加载测试数据
    test_data_pos = np.load('./output1/test_1.npy')
    test_data_neg = np.load('./output1/test_0.npy')
    # 将训练集负样本随机打乱
    np.random.shuffle(test_data_neg)
    # 对测试集负样本进行下采样，使正负样本比例为 1: 1
    sp_test_data_neg = test_data_neg[:len(test_data_pos)]
    test_data = np.vstack((test_data_pos, sp_test_data_neg))
    test_label = np.array([1.0] * len(test_data_pos) + [0.0] * len(sp_test_data_neg))

    return train_data, train_label, test_data, test_label


def params_control(args):
    params = {'clf': args.clf, 'top_n': args.top_n, 'metric': args.metric}
    if 'svm' in args.clf:
        params['cost'], params['gamma'] = utils_params.svm_params_check(args.cost, args.gamma, args.gs_mode)
    if 'rf' in args.clf:
        params['rf_tree'] = utils_params.rf_params_check(args.rf_tree, args.gs_mode)
    if 'nb' in args.clf:
        params['nb_alpha'] = utils_params.nb_params_check(args.nb_alpha, args.gs_mode)
    if 'lgb' in args.clf:
        params['boosting_type'] = args.boosting_type
        params['lgb_tree'], params['num_leaves'] = utils_params.lgb_params_check(args.lgb_tree, args.num_leaves,
                                                                                 args.gs_mode)
    if 'mlp' in args.clf:
        # 不推荐进行遍历
        params['act'] = args.act
        params['hls'] = args.hls
    return params


def clf_layer(ben_data, ben_label, ind_data, ind_label, args):
    train_x, val_x, train_y, val_y = train_test_split(ben_data, ben_label, test_size=0.2, random_state=42)
    train_x, train_y = TomekLinks().fit_sample(train_x, train_y)
    val_x, val_y = TomekLinks().fit_sample(val_x, val_y)
    test_x, test_y = TomekLinks().fit_sample(ind_data, ind_label)
    print('Shape of train data: ', train_x.shape)
    print('Shape of val data: ', val_x.shape)
    print('Shape of test data: ', test_x.shape)
    params = params_control(args)

    metric_lists = []
    prob_lists = []
    if 'svm' in args.clf:
        opt_hps, scores = svm_train(train_x, train_y, val_x, val_y, params)
        svm_metric_list, svm_test_prob_list = svm_predict(train_x, train_y, test_x, test_y, opt_hps, params)
        metric_lists.extend(svm_metric_list)
        prob_lists.extend(svm_test_prob_list)
        print(prob_lists)
    exit()
    if 'rf' in args.clf:
        opt_hps, scores = rf_train(train_x, train_y, val_x, val_y, params)
        rf_metric_list, rf_test_prob_list = rf_predict(train_x, train_y, test_x, test_y, opt_hps, params)
        metric_lists.extend(rf_metric_list)
        prob_lists.extend(rf_test_prob_list)
    if 'nb' in args.clf:
        opt_hps, scores = nb_train(train_x, train_y, val_x, val_y, params)
        nb_metric_list, nb_test_prob_list = nb_predict(train_x, train_y, test_x, test_y, opt_hps, params)
        metric_lists.extend(nb_metric_list)
        prob_lists.extend(nb_test_prob_list)
    if 'lgb' in args.clf:
        opt_hps, scores = lgb_train(train_x, train_y, val_x, val_y, params)
        lgb_metric_list, lgb_test_prob_list = lgb_predict(train_x, train_y, test_x, test_y, opt_hps, params)
        metric_lists.extend(lgb_metric_list)
        prob_lists.extend(lgb_test_prob_list)
    if 'mlp' in args.clf:
        opt_hps, scores = mlp_train(train_x, train_y, val_x, val_y, params)
        mlp_metric_list, mlp_test_prob_list = mlp_predict(train_x, train_y, test_x, test_y, opt_hps, params)
        metric_lists.extend(mlp_metric_list)
        prob_lists.extend(mlp_test_prob_list)

    return metric_lists, prob_lists


def main(args):
    ben_data, ben_label, ind_data, ind_label = load_dataset()
    metric_lists, prob_lists = clf_layer(ben_data, ben_label, ind_data, ind_label, args)
    np.save('./output/score.npy', np.array(prob_lists))


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='Sesica', description="Step into analysis, please select parameters ")

    parse.add_argument('-clf', type=str, nargs='*', choices=['svm', 'rf', 'nb', 'lgb', 'mlp'], required=True,
                       help="The machine learning algorithm, for example: Support Vector Machine(svm).")

    parse.add_argument('-gs_mode', type=int, choices=[0, 1, 2], default=0,
                       help="grid = 0 for no grid search, 1 for rough grid search, 2 for meticulous grid search.")
    parse.add_argument('-metric', type=str, choices=['acc', 'mcc', 'auc', 'b_acc', 'f1'], default='acc',
                       help="The metric for parameter selection")
    parse.add_argument('-top_n', type=int, nargs='*', default=[1, 1, 1, 1, 1],
                       help="Select the n best scores for specific metric.")
    # parameters for svm
    parse.add_argument('-cost', type=int, default=[0], nargs='*', help="Regularization parameter of 'SVM'.")
    parse.add_argument('-gamma', type=int, default=[1], nargs='*', help="Kernel coefficient for 'rbf' of 'SVM'.")
    # parameters for rf
    parse.add_argument('-rf_tree', type=int, default=[100], nargs='*',
                       help="The number of trees in the forest for 'RF'.")
    # parameters for nb
    parse.add_argument('-nb_model', choices=['GaussianNB', 'MultinomialNB', 'BernoulliNB'], default='MultinomialNB',
                       help="The models for Naive Bayes classifier.")
    parse.add_argument('-nb_alpha', type=float, nargs='*', default=[1.0],
                       help="The Additive (Laplace/Lidstone) smoothing parameter for Naive Bayes classifier.")
    # parameters for lgb
    parse.add_argument('-boosting_type', choices=['gbdt', 'dart', 'goss'], default='gbdt',
                       help="'gbdt', traditional Gradient Boosting Decision Tree;"
                            "'dart', Dropouts meet Multiple Additive Regression Trees;"
                            "'goss', Gradient-based One-Side Sampling")
    parse.add_argument('-lgb_tree', type=int, default=[100], nargs='*', help="Number of boosted trees to fit.")
    parse.add_argument('-num_leaves', type=int, default=[31], nargs='*', help="Maximum tree leaves for base learners.")
    # parameters for mlp
    parse.add_argument('-act', choices=['logistic', 'tanh', 'relu'], default='relu',
                       help="Activation function for the hidden layer.")
    parse.add_argument('-hls', type=int, default=[100], nargs='*',
                       help="Hidden layer sizes. The ith element represents the number of neurons in the ith hidden "
                            "layer.")

    argv = parse.parse_args()
    main(argv)
