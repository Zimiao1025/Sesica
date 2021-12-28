from collections import defaultdict

import numpy as np

# from clf import clf_process
from integrate import integration
from utils import util_data, util_ctrl


def load_homo_encodings(vec_file):
    encodings = np.loadtxt(vec_file)
    return encodings


def load_hetero_encodings(vec_file):
    a_encodings = np.loadtxt(vec_file[0])
    b_encodings = np.loadtxt(vec_file[1])
    return a_encodings, b_encodings


def load_connections(label_file):
    pos_pairs = np.loadtxt(label_file[0], dtype=int)
    neg_pairs = np.loadtxt(label_file[1], dtype=int)
    return pos_pairs, neg_pairs


def unpack_associations(pos_pairs, neg_pairs):
    # count the index list of a
    index_list = []
    pos_dict = defaultdict(list)
    for pos_pair in pos_pairs:
        a_index, b_index = pos_pair
        index_list.append(a_index)
        pos_dict[a_index].append(b_index)
    neg_dict = defaultdict(list)

    for neg_pair in neg_pairs:
        a_index, b_index = neg_pair
        index_list.append(a_index)
        neg_dict[a_index].append(b_index)
    index_list = list(set(index_list))

    associations = {}
    for index in index_list:
        pos_connect = pos_dict[index] if index in pos_dict else []
        neg_connect = neg_dict[index] if index in neg_dict else []
        associations[index] = (pos_connect, neg_connect)

    return np.array(index_list, dtype=int), associations


def data_clf_train(index_arr, sp_associations, a_encodings, b_encodings):
    vectors = []
    labels = []
    length = len(index_arr)
    for i in range(length):
        index = index_arr[i]
        for pos_index in sp_associations[index][0]:
            vec = np.hstack((a_encodings[index], b_encodings[pos_index]))
            vectors.append(vec)
            labels.append(1.0)
        for neg_index in sp_associations[index][1]:
            vec = np.hstack((a_encodings[index], b_encodings[neg_index]))
            vectors.append(vec)
            labels.append(0.0)
    return np.array(vectors, dtype=float), np.array(labels, dtype=float)


def data_clf_valid(index_arr, sp_associations, a_encodings, b_encodings):
    vectors = []
    labels = []
    groups = []
    length = len(index_arr)
    for i in range(length):
        index = index_arr[i]
        for pos_index in sp_associations[index][0]:
            vec = np.hstack((a_encodings[index], b_encodings[pos_index]))
            vectors.append(vec)
            labels.append(1.0)
        for neg_index in sp_associations[index][1]:
            vec = np.hstack((a_encodings[index], b_encodings[neg_index]))
            vectors.append(vec)
            labels.append(0.0)
        groups.append(len(sp_associations[index][0]) + len(sp_associations[index][1]))
    return np.array(vectors, dtype=float), np.array(labels, dtype=float), np.array(groups, dtype=int)


def hetero_bmk(args):
    # heterogeneous graph <-- benchmark dataset
    a_encodings, b_encodings = load_hetero_encodings(args.bmk_vec)
    pos_pairs, neg_pairs = load_connections(args.bmk_label)

    # split dataset for parameter selection
    index_list, associations = unpack_associations(pos_pairs, neg_pairs)
    train_index, valid_index, test_index = util_data.dataset_split(index_list)

    # prepare train dataset and valid dataset for ml or dl
    if args.clf != 'none':
        # # save benchmark dataset
        # bmk_x, bmk_y = data_clf_train(index_list, associations, a_encodings, b_encodings)
        # np.save(args.data_dir + 'bmk_x.npy', bmk_x)
        # np.save(args.data_dir + 'bmk_y.npy', bmk_y)

        # under-sampling for a balanced training set
        sp_associations = util_ctrl.sp_ctrl(associations)
        train_x, train_y = data_clf_train(train_index, sp_associations, a_encodings, b_encodings)
        # validation set (Question: associations or sp_associations?)
        valid_x, valid_y, valid_g = data_clf_valid(valid_index, associations, a_encodings, b_encodings)
        # testing set
        test_x, test_y, test_g = data_clf_valid(test_index, associations, a_encodings, b_encodings)
        # save data for repeating experiment
        # training
        np.save(args.data_dir + 'train_x.npy', train_x)
        np.save(args.data_dir + 'train_y.npy', train_y)
        # validation
        np.save(args.data_dir + 'valid_x.npy', valid_x)
        np.save(args.data_dir + 'valid_y.npy', valid_y)
        np.save(args.data_dir + 'valid_g.npy', valid_g)
        # testing
        np.save(args.data_dir + 'test_x.npy', test_x)
        np.save(args.data_dir + 'test_y.npy', test_y)
        np.save(args.data_dir + 'test_g.npy', test_g)


def main(args):
    print("\n******************************** Analysis ********************************\n")
    args = util_ctrl.path_ctrl(args)
    # hetero_bmk(args)
    params = util_ctrl.params_base(args)
    # clf_process.clf_train(args)
    # clf_process.clf_test(args)
    integration.int_or_rank(args, params)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='Sesica', description="Step into analysis, please select parameters ")

    # parameters for
    parse.add_argument('-bmk_vec', nargs='*', required=True, help="The input feature vector files.")
    parse.add_argument('-bmk_label', nargs='*', required=True,
                       help="The input files for positive and negative associations.")

    parse.add_argument('-clf', type=str, nargs='*',
                       choices=['svm', 'rf', 'ert', 'gnb', 'mnb', 'bnb', 'gbdt', 'dart', 'goss', 'mlp', 'none'],
                       default='none',
                       help="The methods of calculating semantic similarity based on probability distribution:\n"
                            " 'svm' --- Support Vector Machine; 'rf' --- Random Forest;\n"
                            " 'ert' --- extremely randomized tree; 'gnb' --- Gaussian Naive Bayes;\n"
                            " 'mnb' --- Multinomial Naive Bayes; 'bnb' --- Bernoulli Naive Bayes;\n"
                            " 'gbdt' --- traditional Gradient Boosting Decision Tree;\n"
                            " 'dart' --- Dropouts meet Multiple Additive Regression Trees;\n"
                            " 'goss' --- Gradient-based One-Side Sampling;\n"
                            " 'mlp' --- Multi-layer Perceptron.\n"
                       )
    parse.add_argument('-arc', type=str, nargs='*',
                       choices=['svm', 'rf', 'ert', 'gnb', 'mnb', 'bnb', 'gbdt', 'dart', 'goss', 'mlp', 'none'],
                       default='none',
                       help="The methods of calculating semantic similarity based on probability distribution:\n"
                            " 'svm'  --- Support Vector Machine; 'rf' --- Random Forest;\n"
                            " 'ert'  --- extremely randomized tree; 'gnb' --- Gaussian Naive Bayes;\n"
                            " 'mnb'  --- Multinomial Naive Bayes; 'bnb' --- Bernoulli Naive Bayes;\n"
                            " 'gbdt' --- traditional Gradient Boosting Decision Tree;\n"
                            " 'dart' --- Dropouts meet Multiple Additive Regression Trees;\n"
                            " 'goss' --- Gradient-based One-Side Sampling;\n"
                            " 'mlp'  --- Multi-layer Perceptron.\n"
                       )
    # parameters for no grid search
    parse.add_argument('-gs_mode', type=int, choices=[0, 1, 2], default=0,
                       help="grid = 0 for no grid search, 1 for rough grid search, 2 for meticulous grid search.")
    parse.add_argument('-metrics', type=str, nargs='*', choices=['aupr', 'auc', 'ndcg@k', 'roc@k'], default=['aupr'],
                       help="The metrics for parameters selection")
    parse.add_argument('-top_n', type=int, nargs='*', default=[1],
                       help="Select the n best scores for specific metric.")
    # parameters for svm
    parse.add_argument('-svm_c', type=int, default=[0], nargs='*', help="Regularization parameter of 'SVM'.")
    parse.add_argument('-svm_g', type=int, default=[1], nargs='*', help="Kernel coefficient for 'rbf' of 'SVM'.")
    # parameters for rf
    parse.add_argument('-rf_t', type=int, default=[100], nargs='*', help="Number of boosted trees to fit.")
    # parameters for ert
    parse.add_argument('-ert_t', type=int, default=[100], nargs='*', help="Number of boosted trees to fit.")
    # parameters for mnb
    parse.add_argument('-mnb_a', type=float, nargs='*', default=[1.0],
                       help="The Additive (Laplace/Lidstone) smoothing parameter for Naive Bayes classifier.")
    # parameters for bnb
    parse.add_argument('-bnb_a', type=float, nargs='*', default=[1.0],
                       help="The Additive (Laplace/Lidstone) smoothing parameter for Naive Bayes classifier.")
    # parameter for gnb is none!
    # parameters for gbdt
    parse.add_argument('-gbdt_t', type=int, default=[100], nargs='*', help="Number of boosted trees to fit.")
    parse.add_argument('-gbdt_n', type=int, default=[31], nargs='*', help="Maximum tree leaves for base learners.")
    # parameters for dart
    parse.add_argument('-dart_t', type=int, default=[100], nargs='*', help="Number of boosted trees to fit.")
    parse.add_argument('-dart_n', type=int, default=[31], nargs='*', help="Maximum tree leaves for base learners.")
    # parameters for goss
    parse.add_argument('-goss_t', type=int, default=[100], nargs='*', help="Number of boosted trees to fit.")
    parse.add_argument('-goss_n', type=int, default=[31], nargs='*', help="Maximum tree leaves for base learners.")
    # parameters for mlp
    parse.add_argument('-act', type=str, nargs='*', choices=['logistic', 'tanh', 'relu'], default=['relu'],
                       help="Activation function for the hidden layer.")
    parse.add_argument('-hls', default=[(100,)], nargs='*',
                       help="Hidden layer sizes. The ith element represents the number of neurons in the ith hidden "
                            "layer.")
    # parameters for integration
    parse.add_argument('-integrate', type=str, choices=['de', 'ga', 'lr', 'ltr', 'none'], default='none',
                       help="Integrate by:\n"
                            " 'none' --- Without integration, the output is sorted directly according to the metric;\n"
                            " 'de' --- differential evolution; 'ga' --- genetic algorithm;\n"
                            " 'lr' --- logistic regression; 'ltr' --- Learning to rank with LambdaRank.\n"
                       )
    # parameters for de and ga
    parse.add_argument('-size_pop', type=int, default=10, help="Population size for 'DE' or 'GA'.")
    parse.add_argument('-max_iter', type=int, default=100, help="Max iterations for 'DE' or 'GA'.")
    # parameters for lr
    parse.add_argument('-lr_c', type=int, default=[0], nargs='*', help="Regularization parameter of 'LR'.")
    # parameters for ltr
    parse.add_argument('-ltr_m', type=int, default=[0], nargs='*',
                       help="Maximum tree depth for base learners, <=0 means no limit.")
    parse.add_argument('-ltr_t', type=int, default=[100], nargs='*', help="Number of boosted trees to fit.")
    parse.add_argument('-ltr_n', type=int, default=[31], nargs='*', help="Maximum tree leaves for base learners.")
    argv = parse.parse_args()
    main(argv)
