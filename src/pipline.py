import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks

from clf.svm_impl import svm_train, svm_predict
from clf.rt_impl import rt_train, rt_predict
from clf.nb_impl import nb_train, nb_predict
from clf.lgb_impl import lgb_train, lgb_predict
from clf.mlp_impl import mlp_train, mlp_predict
from utils import util_params


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


def clf_layers(ben_data, ben_label, ind_data, ind_label, args):
    train_x, val_x, train_y, val_y = train_test_split(ben_data, ben_label, test_size=0.2, random_state=42)
    train_x, train_y = TomekLinks().fit_sample(train_x, train_y)
    val_x, val_y = TomekLinks().fit_sample(val_x, val_y)
    test_x, test_y = TomekLinks().fit_sample(ind_data, ind_label)
    print('Shape of train data: ', train_x.shape)
    print('Shape of val data: ', val_x.shape)
    print('Shape of test data: ', test_x.shape)
    params = util_params.clf_params_control(args)

    metric_lists = []
    prob_lists = []
    for clf in args.clf:
        res = clf_layer(clf, train_x, train_y, val_x, val_y, test_x, test_y, params)
        metric_lists.extend(res[0])
        prob_lists.extend(res[1])
    return metric_lists, prob_lists


def clf_layer(clf, train_x, train_y, val_x, val_y, test_x, test_y, params):
    if clf == 'svm':
        opt_hps, scores = svm_train(train_x, train_y, val_x, val_y, params)
        return svm_predict(train_x, train_y, test_x, test_y, opt_hps, params)
    elif clf in ['rf', 'et']:
        opt_hps, scores = rt_train(clf, train_x, train_y, val_x, val_y, params)
        return rt_predict(clf, train_x, train_y, test_x, test_y, opt_hps, params)
    elif clf in ['gnb', 'mnb', 'bnb']:
        opt_hps, scores = nb_train(clf, train_x, train_y, val_x, val_y, params)
        return nb_predict(train_x, train_y, test_x, test_y, opt_hps, params)
    elif clf in ['gbdt', 'dart', 'goss']:
        opt_hps, scores = lgb_train(train_x, train_y, val_x, val_y, params)
        return lgb_predict(clf, train_x, train_y, test_x, test_y, opt_hps, params)
    else:
        opt_hps, scores = mlp_train(train_x, train_y, val_x, val_y, params)
        return mlp_predict(train_x, train_y, test_x, test_y, opt_hps, params)


def main(args):
    ben_data, ben_label, ind_data, ind_label = load_dataset()
    metric_lists, prob_lists = clf_layers(ben_data, ben_label, ind_data, ind_label, args)
    np.save('./output/score.npy', np.array(prob_lists))


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='Sesica', description="Step into analysis, please select parameters ")

    parse.add_argument('-score', type=str, nargs='*', choices=['ED', 'MD', 'CD', 'HD', 'JSC', 'CS', 'PCC', 'KLD'],
                       help="Choose whether calculate semantic similarity score for feature vectors.")

    parse.add_argument('-clf', type=str, nargs='*',
                       choices=['svm', 'rf', 'et', 'gnb', 'mnb', 'bnb', 'gbdt', 'dart', 'goss', 'mlp'],
                       help="The machine learning algorithm, for example: Support Vector Machine(svm).")
    # parameters for no grid search
    parse.add_argument('-gs_mode', type=int, choices=[0, 1, 2], default=0,
                       help="grid = 0 for no grid search, 1 for rough grid search, 2 for meticulous grid search.")
    parse.add_argument('-metric', type=str, choices=['acc', 'mcc', 'auc', 'b_acc', 'f1'], default='acc',
                       help="The metric for parameter selection")
    parse.add_argument('-top_n', type=int,
                       help="Select the n best scores for specific metric.")
    # parameters for svm
    parse.add_argument('-cost', type=int, default=[0], nargs='*', help="Regularization parameter of 'SVM'.")
    parse.add_argument('-gamma', type=int, default=[1], nargs='*', help="Kernel coefficient for 'rbf' of 'SVM'.")
    # parameters for rf and et
    parse.add_argument('-n_estimators', type=int, default=[100], nargs='*',
                       help="The number of trees in the forest.")
    # parameters for nb
    parse.add_argument('-nb_alpha', type=float, nargs='*', default=[1.0],
                       help="The Additive (Laplace/Lidstone) smoothing parameter for Naive Bayes classifier.")
    # parameters for lgb
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
