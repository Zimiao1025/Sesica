import numpy as np
import pandas as pd

from arc import arc_process
from utils import util_data, util_ctrl, util_graph, util_word


def data_arc_train(index_arr, associations, a_encodings, fixed_len):
    # ,id_left,text_left,length_left,id_right,text_right,length_right,label
    length = len(index_arr)
    data_dict_list = []
    for i in range(length):
        index = index_arr[i]
        id_left = 'Q' + str(index)
        for pos_index in associations[index][0]:
            id_right = id_left + '-' + str(pos_index)
            tmp_dict = {'id_left': id_left, 'text_left': a_encodings[index], 'length_left': fixed_len,
                        'id_right': id_right, 'text_right': a_encodings[pos_index], 'length_right': fixed_len,
                        'label': 1.0}
            data_dict_list.append(tmp_dict)
        for neg_index in associations[index][1]:
            id_right = id_left + '-' + str(neg_index)
            tmp_dict = {'id_left': id_left, 'text_left': a_encodings[index], 'length_left': fixed_len,
                        'id_right': id_right, 'text_right': a_encodings[neg_index], 'length_right': fixed_len,
                        'label': 0.0}
            data_dict_list.append(tmp_dict)
    return pd.DataFrame(data_dict_list)


def data_arc_valid(index_arr, associations, a_encodings, fixed_len):
    # ,id_left,text_left,length_left,id_right,text_right,length_right,label
    length = len(index_arr)
    data_dict_list = []
    groups = []
    for i in range(length):
        index = index_arr[i]
        id_left = 'Q' + str(index)
        for pos_index in associations[index][0]:
            id_right = id_left + '-' + str(pos_index)
            tmp_dict = {'id_left': id_left, 'text_left': a_encodings[index], 'length_left': fixed_len,
                        'id_right': id_right, 'text_right': a_encodings[pos_index], 'length_right': fixed_len,
                        'label': 1.0}
            data_dict_list.append(tmp_dict)
        for neg_index in associations[index][1]:
            id_right = id_left + '-' + str(neg_index)
            tmp_dict = {'id_left': id_left, 'text_left': a_encodings[index], 'length_left': fixed_len,
                        'id_right': id_right, 'text_right': a_encodings[neg_index], 'length_right': fixed_len,
                        'label': 0.0}
            data_dict_list.append(tmp_dict)
        groups.append(len(associations[index][0]) + len(associations[index][1]))
    return pd.DataFrame(data_dict_list), np.array(groups, dtype=int)


def arc_bmk(args):
    # heterogeneous graph <-- benchmark dataset
    word_encoding = util_word.km_words(args.bmk_fasta, args.args.alphabet, args.fixed_len, args.word_size)
    pos_pairs, neg_pairs = util_graph.load_connections(args.bmk_label)

    # split dataset for parameter selection
    index_list, associations = util_graph.unpack_associations(pos_pairs, neg_pairs)
    train_index, valid_index, test_index = util_data.dataset_split(index_list)

    if args.arc != 'none':
        train_df = data_arc_train(train_index, associations, word_encoding, args.fixed_len)
        valid_df, valid_g = data_arc_valid(valid_index, associations, word_encoding, args.fixed_len)
        test_df, test_g = data_arc_valid(test_index, associations, word_encoding, args.fixed_len)
        # save data for repeating experiment
        train_df.to_csv(args.data_dir + 'train_df.csv')
        valid_df.to_csv(args.data_dir + 'valid_df.csv')
        test_df.to_csv(args.data_dir + 'test_df.csv')
        # save group for ltr
        np.save(args.data_dir + 'valid_g.npy', valid_g)
        np.save(args.data_dir + 'test_g.npy', test_g)


def main(args):
    print("\n******************************** Analysis ********************************\n")
    args = util_ctrl.path_ctrl(args)
    arc_bmk(args)
    params = util_ctrl.params_base(args)
    # print(params)
    arc_process.arc_ctrl(args, params)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='Sesica', description="Step into analysis, please select parameters ")

    # parameters for input
    parse.add_argument('-bmk_fasta', nargs='*', required=True, help="The input sequence files in fasta format.")
    parse.add_argument('-bmk_label', nargs='*', required=True,
                       help="The input files for positive and negative associations.")
    parse.add_argument('-ind', choices=[True, False], default=False,
                       help="The input files for positive and negative associations.")
    parse.add_argument('-arc', type=str, nargs='*',
                       choices=['arci', 'arcii', 'dssm', 'cdssm', 'drmm', 'none'],
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
    parse.add_argument('-metrics', type=str, nargs='*', choices=['aupr', 'auc', 'ndcg', 'ndcg@1', 'roc@1', 'ndcg@5',
                                                                 'roc@5', 'ndcg@10', 'roc@10', 'ndcg@20', 'roc@20',
                                                                 'ndcg@50', 'roc@50'], default=['aupr'],
                       help="The metrics for parameters selection")
    parse.add_argument('-top_n', type=int, nargs='*', default=[1],
                       help="Select the n best scores for specific metric.")
    # parameters for arc
    parse.add_argument('-num_neg', type=int, nargs='*', default=[4], help="Number of negative samples for ranking.")
    parse.add_argument('-epoch', type=int, nargs='*', default=[10], help="Number of epochs for training.")
    parse.add_argument('-emb_in', type=int, default=[10000, 30000], nargs='*', help="Embedding size input.")
    parse.add_argument('-arci_e', type=int, default=30001, help="Embedding size input for arci model.")
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
