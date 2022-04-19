import numpy as np

from clf import clf_process
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


def data_clf_score(a_encodings, b_encodings, tag='A'):
    fst_len = len(a_encodings) if tag == 'A' else len(b_encodings)
    snd_len = len(b_encodings) if tag == 'A' else len(a_encodings)
    vectors = []
    score_labels = []
    for i in range(fst_len):
        for j in range(snd_len):
            if tag == 'A':
                vec = np.hstack((a_encodings[i], b_encodings[j]))
            else:
                vec = np.hstack((a_encodings[j], b_encodings[i]))
            vectors.append(vec)
            # [query_idx, doc_idx]
            score_labels.append([i, j])
    return np.array(vectors, dtype=float), np.array(score_labels, dtype=int)


def clf_bmk_hetero(args):
    # heterogeneous graph <-- benchmark dataset
    a_df, a_encodings = util_graph.load_vec_encodings(args.bmk_vec_a)
    b_df, b_encodings = util_graph.load_vec_encodings(args.bmk_vec_b)
    a_df.to_csv(args.data_dir + 'bmk_vec_a.csv')
    b_df.to_csv(args.data_dir + 'bmk_vec_b.csv')
    pos_pairs, neg_pairs = util_graph.load_connections(args.bmk_label)

    # split dataset for parameter selection
    index_list, associations = util_graph.unpack_associations(pos_pairs, neg_pairs)
    train_index, valid_index, test_index = util_data.dataset_split(index_list)

    # prepare train dataset and valid dataset for ml or dl

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


def clf_bmk_homo(args):
    # heterogeneous graph <-- benchmark dataset
    a_df, a_encodings = util_graph.load_vec_encodings(args.bmk_vec)
    a_df.to_csv(args.data_dir + 'bmk_vec.csv')
    pos_pairs, neg_pairs = util_graph.load_connections(args.bmk_label)

    # split dataset for parameter selection
    index_list, associations = util_graph.unpack_associations(pos_pairs, neg_pairs)
    train_index, valid_index, test_index = util_data.dataset_split(index_list)

    # prepare train dataset and valid dataset for ml or dl
    # under-sampling for a balanced training set
    sp_associations = util_ctrl.sp_ctrl(associations)
    train_x, train_y = data_clf_train(train_index, sp_associations, a_encodings, a_encodings)
    # validation set (Question: associations or sp_associations?)
    valid_x, valid_y, valid_g = data_clf_valid(valid_index, associations, a_encodings, a_encodings)
    # testing set
    test_x, test_y, test_g = data_clf_valid(test_index, associations, a_encodings, a_encodings)
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


def clf_test_hetero(args):
    # heterogeneous graph <-- ind dataset
    a_df, a_encodings = util_graph.load_vec_encodings(args.ind_vec_a)
    b_df, b_encodings = util_graph.load_vec_encodings(args.ind_vec_b)
    a_df.to_csv(args.data_dir + 'ind_vec_a.csv')
    b_df.to_csv(args.data_dir + 'ind_vec_b.csv')
    pos_pairs, neg_pairs = util_graph.load_connections(args.ind_label)

    # split dataset for parameter selection
    index_list, associations = util_graph.unpack_associations(pos_pairs, neg_pairs)

    # under-sampling for a balanced training set
    ind_x, ind_y, ind_g = data_clf_valid(index_list, associations, a_encodings, b_encodings)

    # save data for repeating experiment
    # ind
    np.save(args.data_dir + 'ind_x.npy', ind_x)
    np.save(args.data_dir + 'ind_y.npy', ind_y)
    np.save(args.data_dir + 'ind_g.npy', ind_g)


def clf_test_homo(args):
    # heterogeneous graph <-- ind dataset
    a_df, a_encodings = util_graph.load_vec_encodings(args.ind_vec)
    a_df.to_csv(args.data_dir + 'ind_vec.csv')

    pos_pairs, neg_pairs = util_graph.load_connections(args.ind_label)

    # split dataset for parameter selection
    index_list, associations = util_graph.unpack_associations(pos_pairs, neg_pairs)

    # under-sampling for a balanced training set
    ind_x, ind_y, ind_g = data_clf_valid(index_list, associations, a_encodings, a_encodings)

    # save data for repeating experiment
    # ind
    np.save(args.data_dir + 'ind_x.npy', ind_x)
    np.save(args.data_dir + 'ind_y.npy', ind_y)
    np.save(args.data_dir + 'ind_g.npy', ind_g)


def clf_score_hetero(args):
    # heterogeneous graph <-- ind dataset
    if args.ind_vec_a:
        a_df, a_encodings = util_graph.load_vec_encodings(args.ind_vec_a)
        b_df, b_encodings = util_graph.load_vec_encodings(args.bmk_vec_b)
        a_df.to_csv(args.data_dir + 'ind_vec_a.csv')
        score_x, score_pair = data_clf_score(a_encodings, b_encodings, tag='A')
        np.save(args.data_dir + 'score_x.npy', score_x)
        np.save(args.data_dir + 'score_pair_A.npy', score_pair)
    if args.ind_vec_b:
        a_df, a_encodings = util_graph.load_vec_encodings(args.bmk_vec_a)
        b_df, b_encodings = util_graph.load_vec_encodings(args.ind_vec_b)
        b_df.to_csv(args.data_dir + 'ind_vec_b.csv')
        score_x, score_pair = data_clf_score(a_encodings, b_encodings, tag='B')
        np.save(args.data_dir + 'score_x.npy', score_x)
        np.save(args.data_dir + 'score_pair_B.npy', score_pair)


def clf_score_homo(args):
    # heterogeneous graph <-- ind dataset
    if args.ind_vec:
        a_df, a_encodings = util_graph.load_vec_encodings(args.ind_vec)
        b_df, b_encodings = util_graph.load_vec_encodings(args.bmk_vec)
        a_df.to_csv(args.data_dir + 'ind_vec.csv')
        score_x, score_pair = data_clf_score(a_encodings, b_encodings, tag='A')
        np.save(args.data_dir + 'score_x.npy', score_x)
        np.save(args.data_dir + 'score_pair.npy', score_pair)


def main(args):
    print("\n******************************** Analysis ********************************\n")
    if args.clf != 'none':
        util_ctrl.args_check(args)
        args = util_ctrl.clf_path_ctrl(args)
        params = util_ctrl.params_clf(args)
        # print(params)
        if args.data_type == 'hetero':
            print(' |Heterogeneous| '.center(40, '-'))
            clf_bmk_hetero(args)
            clf_process.clf_train(args, params)
            clf_process.clf_eval(args, params)
            if args.ind == 'test':
                clf_test_hetero(args)
                clf_process.clf_test(args, params)
            if args.ind == 'score':
                clf_score_hetero(args)
                if args.ind_vec_a:
                    clf_process.clf_score_hetero(args, params, tag='A')
                if args.ind_vec_b:
                    clf_process.clf_score_hetero(args, params, tag='B')
        else:
            print(' |Homogeneous| '.center(40, '-'))
            clf_bmk_homo(args)
            clf_process.clf_train(args, params)
            clf_process.clf_eval(args, params)
            if args.ind == 'test':
                clf_test_homo(args)
                clf_process.clf_test(args, params)
            if args.ind == 'score':
                clf_score_homo(args)
                clf_process.clf_score_homo(args, params)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='Sesica', description="Step into analysis, please select parameters ")

    # parameters for clf
    parse.add_argument('-base_dir', required=True, help="The relative path or absolute path to store result.")
    parse.add_argument('-data_type', required=True, choices=['hetero', 'homo'],
                       help="Select the input heterogeneous data or homogeneous data.")
    parse.add_argument('-bmk_vec_a', help="The feature vector files of heterogeneous benchmark datasets A.")
    parse.add_argument('-bmk_vec_b', help="The feature vector files of heterogeneous benchmark datasets B.")
    parse.add_argument('-bmk_vec', help="The feature vector files of homogeneous benchmark datasets.")
    parse.add_argument('-bmk_label', nargs='*', required=True,
                       help="The input files for positive and negative associations of benchmark datasets."
                            "File of positive associations should input before file of negative associations.")
    parse.add_argument('-ind', choices=['test', 'score', 'none'], default='none',
                       help="Input independent test dataset or not. "
                            "If you select 'test', you have to input vector file and label file; "
                            "If you select 'score', you only need to input vector file, then the similarities score "
                            "between benchmark dataset and independent dataset will be calculated. ")
    parse.add_argument('-ind_vec_a', help="The feature vector files of heterogeneous independent datasets A.")
    parse.add_argument('-ind_vec_b', help="The feature vector files of heterogeneous independent datasets B.")
    parse.add_argument('-ind_vec', help="The feature vector files of homogeneous independent datasets.")
    parse.add_argument('-ind_label', nargs='*',
                       help="The input files for positive and negative associations of independent datasets."
                       "File of positive associations should input before file of negative associations.")
    parse.add_argument('-clf', type=str, nargs='*',
                       choices=['svm', 'rf', 'ert', 'knn', 'mnb', 'gbdt', 'dart', 'goss', 'mlp', 'none'],
                       default='none',
                       help="The methods of calculating semantic similarity based on probability distribution:\n"
                            " 'svm' --- Support Vector Machine with RBF kernel;\n"
                            " 'rf' --- Random Forest;\n"
                            " 'ert' --- extremely randomized tree;\n"
                            " 'knn' --- k-nearest neighbors vote;\n"
                            " 'mnb' ---Multinomial Naive Bayes;\n"
                            " 'gbdt' --- traditional Gradient Boosting Decision Tree;\n"
                            " 'dart' --- Dropouts meet Multiple Additive Regression Trees;\n"
                            " 'goss' --- Gradient-based One-Side Sampling;\n"
                            " 'mlp' --- Multi-layer Perceptron;\n"
                            " 'none' --- none of model will be selected.\n"
                       )
    parse.add_argument('-scale', type=str, nargs='*', choices=['mms', 'ss', 'nor', 'none'], default=['none'],
                       help=" 'mms' --- scale with MinMaxScaler;\n"
                            " 'ss' --- scale with StandardScaler;\n"
                            " 'nor' --- scale with Normalizer;\n"
                            " 'none'  --- without scale.\n")
    # parameters for no grid search
    parse.add_argument('-gs_mode', type=int, choices=[0, 1, 2], default=0,
                       help="grid = 0 for no grid search, 1 for rough grid search, 2 for meticulous grid search.")
    parse.add_argument('-metric', type=str,
                       choices=['aupr', 'auc', 'ndcg', 'roc@1', 'ndcg@10', 'roc@10', 'ndcg@20', 'roc@20', 'ndcg@50',
                                'roc@50'], default='aupr', help="The metrics used for parameters selection")
    parse.add_argument('-top_n', type=int, nargs='*', default=[1],
                       help="Select the n best models for specific metric of distribution based methods.")
    # parameters for svm
    parse.add_argument('-svm_c', type=int, default=[0], nargs='*', help="Regularization parameter of 'RSVM'.")
    parse.add_argument('-svm_g', type=int, default=[1], nargs='*', help="Kernel coefficient of 'RSVM'.")
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
    argv = parse.parse_args()
    main(argv)
