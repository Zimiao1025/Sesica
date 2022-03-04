import numpy as np
import pandas as pd

from arc import arc_process
# from clf import clf_process
# from integrate import integration
from plot import plot_process
from utils import util_data, util_ctrl, util_graph


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


def data_clf_valid(index_arr, associations, a_encodings, b_encodings):
    vectors = []
    labels = []
    groups = []
    length = len(index_arr)
    for i in range(length):
        index = index_arr[i]
        for pos_index in associations[index][0]:
            vec = np.hstack((a_encodings[index], b_encodings[pos_index]))
            vectors.append(vec)
            labels.append(1.0)
        for neg_index in associations[index][1]:
            vec = np.hstack((a_encodings[index], b_encodings[neg_index]))
            vectors.append(vec)
            labels.append(0.0)
        groups.append(len(associations[index][0]) + len(associations[index][1]))
    return np.array(vectors, dtype=float), np.array(labels, dtype=float), np.array(groups, dtype=int)


def data_arc_train(index_arr, associations, a_encodings, b_encodings):
    # ,id_left,text_left,length_left,id_right,text_right,length_right,label
    left_len = len(a_encodings[0])
    right_len = len(b_encodings[0])
    length = len(index_arr)
    data_dict_list = []
    for i in range(length):
        index = index_arr[i]
        id_left = 'Q' + str(index)
        for pos_index in associations[index][0]:
            id_right = id_left + '-' + str(pos_index)
            tmp_dict = {'id_left': id_left, 'text_left': a_encodings[index], 'length_left': left_len,
                        'id_right': id_right, 'text_right': b_encodings[pos_index], 'length_right': right_len,
                        'label': 1.0}
            data_dict_list.append(tmp_dict)
        for neg_index in associations[index][1]:
            id_right = id_left + '-' + str(neg_index)
            tmp_dict = {'id_left': id_left, 'text_left': a_encodings[index], 'length_left': left_len,
                        'id_right': id_right, 'text_right': b_encodings[neg_index], 'length_right': right_len,
                        'label': 0.0}
            data_dict_list.append(tmp_dict)
    return pd.DataFrame(data_dict_list)


def data_arc_valid(index_arr, associations, a_encodings, b_encodings):
    # ,id_left,text_left,length_left,id_right,text_right,length_right,label
    left_len = len(a_encodings[0])
    right_len = len(b_encodings[0])
    length = len(index_arr)
    data_dict_list = []
    groups = []
    for i in range(length):
        index = index_arr[i]
        id_left = 'Q' + str(index)
        for pos_index in associations[index][0]:
            id_right = id_left + '-' + str(pos_index)
            tmp_dict = {'id_left': id_left, 'text_left': a_encodings[index], 'length_left': left_len,
                        'id_right': id_right, 'text_right': b_encodings[pos_index], 'length_right': right_len,
                        'label': 1.0}
            data_dict_list.append(tmp_dict)
        for neg_index in associations[index][1]:
            id_right = id_left + '-' + str(neg_index)
            tmp_dict = {'id_left': id_left, 'text_left': a_encodings[index], 'length_left': left_len,
                        'id_right': id_right, 'text_right': b_encodings[neg_index], 'length_right': right_len,
                        'label': 0.0}
            data_dict_list.append(tmp_dict)
        groups.append(len(associations[index][0]) + len(associations[index][1]))
    return pd.DataFrame(data_dict_list), np.array(groups, dtype=int)


def clf_bmk(args):
    # heterogeneous graph <-- benchmark dataset
    a_encodings, b_encodings = util_graph.load_hetero_encodings(args.bmk_vec)
    pos_pairs, neg_pairs = util_graph.load_connections(args.bmk_label)

    # split dataset for parameter selection
    index_list, associations = util_graph.unpack_associations(pos_pairs, neg_pairs)
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
    clf_bmk(args)
    params = util_ctrl.params_base(args)
    # print(params)
    arc_process.arc_ctrl(args, params)
    # clf_process.clf_train(args, params)
    # clf_process.clf_test(args, params)
    # integration.int_or_rank(args, params)
    plot_process.plot_fig(args, False, params)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='Sesica', description="Step into analysis, please select parameters ")

    # parameters for
    parse.add_argument('-bmk_vec', nargs='*', required=True, help="The input feature vector files.")
    parse.add_argument('-bmk_fasta', nargs='*', required=True, help="The input sequence files in fasta format.")
    parse.add_argument('-bmk_label', nargs='*', required=True,
                       help="The input files for positive and negative associations.")
    parse.add_argument('-ind', choices=[True, False], default=False,
                       help="The input files for positive and negative associations.")
    parse.add_argument('-clf', type=str, nargs='*',
                       choices=['svm', 'rf', 'ert', 'knn', 'mnb', 'gbdt', 'dart', 'goss', 'mlp', 'none'],
                       default='none',
                       help="The methods of calculating semantic similarity based on probability distribution:\n"
                            " 'svm' --- Support Vector Machine with RBF kernel; 'rf' --- Random Forest;\n"
                            " 'ert' --- extremely randomized tree; 'knn' --- k-nearest neighbors vote;\n"
                            " 'mnb' ---Multinomial Naive Bayes;\n"
                            " 'gbdt' --- traditional Gradient Boosting Decision Tree;\n"
                            " 'dart' --- Dropouts meet Multiple Additive Regression Trees;\n"
                            " 'goss' --- Gradient-based One-Side Sampling;\n"
                            " 'mlp' --- Multi-layer Perceptron.\n"
                       )
    parse.add_argument('-scale', type=str, nargs='*', choices=['mms', 'ss', 'nor', 'none'], default=['none'],
                       help=" 'mms' --- scale with MinMaxScaler;\n"
                            " 'ss' --- scale with StandardScaler;\n"
                            " 'nor' --- scale with Normalizer;\n"
                            " 'none'  --- without scale.\n")
    # parameters for no grid search
    parse.add_argument('-gs_mode', type=int, choices=[0, 1, 2], default=0,
                       help="grid = 0 for no grid search, 1 for rough grid search, 2 for meticulous grid search.")
    parse.add_argument('-metrics', type=str, nargs='*', choices=['aupr', 'auc', 'ndcg', 'ndcg@1', 'roc@1', 'ndcg@5',
                                                                 'roc@5', 'ndcg@10', 'roc@10', 'ndcg@20', 'roc@20',
                                                                 'ndcg@50', 'roc@50'], default=['aupr'],
                       help="The metrics for parameters selection")
    parse.add_argument('-top_n', type=int, nargs='*', default=[1],
                       help="Select the n best scores for specific metric.")
    # parameters for svm
    parse.add_argument('-svm_c', type=int, default=[0], nargs='*', help="Regularization parameter of 'RSVM'.")
    parse.add_argument('-svm_g', type=int, default=[1], nargs='*', help="Kernel coefficient for 'rbf' of 'RSVM'.")
    # parameters for rf
    parse.add_argument('-rf_t', type=int, default=[100], nargs='*', help="Number of boosted trees to fit.")
    # parameters for ert
    parse.add_argument('-ert_t', type=int, default=[100], nargs='*', help="Number of boosted trees to fit.")
    # parameters for mnb
    parse.add_argument('-mnb_a', type=float, nargs='*', default=[1.0],
                       help="The Additive (Laplace/Lidstone) smoothing parameter for Naive Bayes classifier.")
    # parameters for knn
    parse.add_argument('-knn_n', type=int, nargs='*', default=[100],
                       help="Number of neighbors to use by default for k-neighbors queries.")
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
    parse.add_argument('-act', type=str, choices=['logistic', 'tanh', 'relu'], default='relu',
                       help="Activation function for the hidden layer.")
    parse.add_argument('-hls', type=int, default=[100], nargs='*',
                       help="Hidden layer sizes. The ith element represents the number of neurons in the ith hidden "
                            "layer.")
    # parameters for integration
    parse.add_argument('-integrate', type=str, choices=['ltr', 'none'], default='none',
                       help="Integrate by:\n"
                            " 'none' --- Without integration, the output is sorted directly according to the metric;\n"
                            " 'ltr' --- Learning to rank with LambdaRank.\n"
                       )
    # parameters for ltr
    parse.add_argument('-ltr_m', type=int, default=[0], nargs='*',
                       help="Maximum tree depth for base learners, <=0 means no limit.")
    parse.add_argument('-ltr_t', type=int, default=[100], nargs='*', help="Number of boosted trees to fit.")
    parse.add_argument('-ltr_n', type=int, default=[31], nargs='*', help="Maximum tree leaves for base learners.")
    parse.add_argument('-plot', type=str, choices=['prc', 'roc', 'box', 'polar', 'hp', '3d', 'dist', 'pie', 'bar',
                                                   'none'], default='none', nargs='*',
                       help="Integrate by:\n"
                            " 'none' --- Don't plot;\n"
                            " 'prc' --- precision-recall Curve; 'roc' --- receiver operating characteristic;\n"
                            " 'box' --- box figure for evaluation results; 'hp' --- heat map of the relevance.\n"
                            " '3d' --- 3d figure for dimension reduce; 'dist' --- histogram for distribution.\n"
                            " 'pie' --- pie figure for optimal weight; 'bar' --- histogram for feature importance.\n"
                       )
    parse.add_argument('-plot_metric', type=str, choices=['aupr', 'auc', 'ndcg@k', 'roc@k', 'metric_1'],
                       default='metric_1', help="The metrics for plot, the -plot_metric should be a metric included in "
                                                "-metric parameter you chose before. The metric_1 means the first "
                                                "metric you chose in -metrics parameter")
    argv = parse.parse_args()
    main(argv)
