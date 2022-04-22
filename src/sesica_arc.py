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
    labels = []
    groups = []
    for i in range(length):
        index = index_arr[i]
        id_left = 'Q' + str(index)
        for pos_index in associations[index][0]:
            id_right = id_left + '-' + str(pos_index)
            tmp_dict = {'id_left': id_left, 'text_left': a_encodings[index], 'length_left': fixed_len,
                        'id_right': id_right, 'text_right': a_encodings[pos_index], 'length_right': fixed_len,
                        'label': 1.0}
            labels.append(1.0)
            data_dict_list.append(tmp_dict)
        for neg_index in associations[index][1]:
            id_right = id_left + '-' + str(neg_index)
            tmp_dict = {'id_left': id_left, 'text_left': a_encodings[index], 'length_left': fixed_len,
                        'id_right': id_right, 'text_right': a_encodings[neg_index], 'length_right': fixed_len,
                        'label': 0.0}
            labels.append(0.0)
            data_dict_list.append(tmp_dict)
        groups.append(len(associations[index][0]) + len(associations[index][1]))
    return pd.DataFrame(data_dict_list), np.array(labels), np.array(groups, dtype=int)


def arc_bmk(args):
    # heterogeneous graph <-- benchmark dataset
    word_encoding = util_word.km_words(args.bmk_fasta, args.category, args.fixed_len, args.word_size)
    print('Shape of word_encoding: ', word_encoding.shape)
    np.save(args.data_dir + 'bmk_encoding.npy', word_encoding)
    pos_pairs, neg_pairs = util_graph.load_connections(args.bmk_label)

    # split dataset for parameter selection
    index_list, associations = util_graph.unpack_associations(pos_pairs, neg_pairs)
    train_index, valid_index, test_index = util_data.dataset_split(index_list)

    if args.arc != 'none':
        train_df = data_arc_train(train_index, associations, word_encoding, args.fixed_len)
        valid_df, valid_y, valid_g = data_arc_valid(valid_index, associations, word_encoding, args.fixed_len)
        test_df, test_y, test_g = data_arc_valid(test_index, associations, word_encoding, args.fixed_len)
        # save data for repeating experiment
        train_df.to_csv(args.data_dir + 'train_df.csv')
        valid_df.to_csv(args.data_dir + 'valid_df.csv')
        test_df.to_csv(args.data_dir + 'test_df.csv')
        # save group for ltr
        np.save(args.data_dir + 'valid_y.npy', valid_y)
        np.save(args.data_dir + 'test_y.npy', test_y)
        np.save(args.data_dir + 'valid_g.npy', valid_g)
        np.save(args.data_dir + 'test_g.npy', test_g)


def main(args):
    print("\n******************************** Analysis ********************************\n")
    args = util_ctrl.arc_path_ctrl(args)
    arc_bmk(args)
    params = util_ctrl.params_arc(args)
    # print(params)
    arc_process.arc_ctrl(args, params)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='Sesica', description="Step into analysis, please select parameters ")

    # parameters for arc
    parse.add_argument('-base_dir', required=True, help="The relative path or absolute path to store result.")
    parse.add_argument('-bmk_fasta', required=True, help="The input sequence files in fasta format of benchmark datasets.")
    parse.add_argument('-bmk_label', nargs='*', required=True,
                       help="The input files for positive and negative associations of benchmark datasets.")
    parse.add_argument('-ind', choices=[True, False], default=False, help="Input independent test dataset or not.")
    parse.add_argument('-ind_fasta', help="The input sequence files in fasta format of independent datasets.")
    parse.add_argument('-ind_label', nargs='*',
                       help="The input files for positive and negative associations of independent datasets.")
    parse.add_argument('-category', type=str, choices=['DNA', 'RNA', 'Protein'], required=True,
                       help="The category of input sequences.")
    parse.add_argument('-word_size', type=int, default=4, help="The kmer word size for making vocabulary.")
    parse.add_argument('-fixed_len', type=int, default=500,
                       help="The length of sequence will be fixed via cutting or padding.")
    parse.add_argument('-arc', type=str, nargs='*',
                       choices=['arci', 'arcii', 'dssm', 'cdssm', 'drmm', 'drmmtks', 'match_lstm', 'duet', 'knrm',
                                'conv_knrm', 'esim', 'bimpm', 'match_pyramid', 'match_srnn', 'anmm', 'mv_lstm', 'diin',
                                'hbmp', 'none'],
                       default='none',
                       help="The methods of calculating semantic similarity based on probability distribution:\n"
                            " 'arci'  --- this model is an implementation of Convolutional Neural Network Architectures for Matching Natural Language Sentences;\n"
                            " 'arcii'  --- this model is an implementation of Convolutional Neural Network Architectures for Matching Natural Language Sentences;\n"
                            " 'dssm'  --- this model is an implementation of Learning Deep Structured Semantic Models for Web Search using Clickthrough Data;\n"
                            " 'cdssm' --- this model is an implementation of Learning Semantic Representations Using Convolutional Neural Networks for Web Search;\n"
                            " 'drmm' --- this model is an implementation of A Deep Relevance Matching Model for Ad-hoc Retrieval;\n"
                            " 'drmmtks' --- this model is an implementation of A Deep Top-K Relevance Matching Model for Ad-hoc Retrieval;\n"
                            " 'match_lstm'  --- this model is an implementation of Machine Comprehension Using Match-LSTM and Answer Pointer;\n"
                            " 'duet'  --- this model is an implementation of Learning to Match Using Local and Distributed Representations of Text for Web Search;\n"
                            " 'knrm'  --- this model is an implementation of End-to-End Neural Ad-hoc Ranking with Kernel Pooling;\n"
                            " 'conv_knrm'  --- this model is an implementation of Convolutional neural networks for soft-matching n-grams in ad-hoc search;\n"
                            " 'esim' --- this model is an implementation of Enhanced LSTM for Natural Language Inference;\n"
                            " 'bimpm' --- this model is an implementation of Bilateral Multi-Perspective Matching for Natural Language Sentences;\n"
                            " 'match_pyramid' --- this model is an implementation of Text Matching as Image Recognition;\n"
                            " 'match_srnn'  --- this model is an implementation of Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN;\n"
                            " 'anmm'  --- this model is an implementation of aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model;\n"
                            " 'mv_lstm' --- this model is an implementation of A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations;\n"
                            " 'diin' --- this model is an implementation of Natural Language Inference Over Interaction Space;\n"
                            " 'hbmp' --- this model is an implementation of Sentence Embeddings in NLI with Iterative Refinement Encoders;\n"
                            " 'none' --- none of model will be selected..\n"
                       )
    parse.add_argument('-metric', type=str,
                       choices=['aupr', 'auc', 'ndcg', 'roc@1', 'ndcg@10', 'roc@10', 'ndcg@20', 'roc@20', 'ndcg@50',
                                'roc@50'], default='aupr', help="The metrics used for parameters selection")
    # parameters for arci
    parse.add_argument('-arci_neg', type=int, default=4, help="Number of negative samples for ranking of arci model.")
    parse.add_argument('-arci_epoch', type=int, default=5, help="Number of epochs for training of arci model.")
    parse.add_argument('-arci_dropout', type=float, default=0.5, help="Dropout rate for training of arci model.")
    parse.add_argument('-arci_lr', type=float, default=3e-4, help="Learning rate for optimizer of arci model.")
    parse.add_argument('-arci_emb', type=int, default=128, help="Embedding output dimension of arci model.")
    parse.add_argument('-arci_layers', type=int, default=1, help="Number of mlp layers for arci model.")
    parse.add_argument('-arci_units', type=int, default=64, help="Number of mlp units for arci model.")
    # parameters for arcii
    parse.add_argument('-arcii_neg', type=int, default=4, help="Number of negative samples for ranking of arcii model.")
    parse.add_argument('-arcii_epoch', type=int, default=5, help="Number of epochs for training of arcii model.")
    parse.add_argument('-arcii_dropout', type=float, default=0.5, help="Dropout rate for training of arcii model.")
    parse.add_argument('-arcii_lr', type=float, default=3e-4, help="Learning rate for optimizer of arcii model.")
    parse.add_argument('-arcii_emb', type=int, default=128, help="Embedding output dimension of arcii model.")
    # parameters for dssm
    parse.add_argument('-dssm_neg', type=int, default=4, help="Number of negative samples for ranking of dssm model.")
    parse.add_argument('-dssm_ngram', type=int, default=3, help="The N of n-gram model for dssm model.")
    parse.add_argument('-dssm_epoch', type=int, default=5, help="Number of epochs for training of dssm model.")
    parse.add_argument('-dssm_dropout', type=float, default=0.5, help="Dropout rate for training of dssm model.")
    parse.add_argument('-dssm_lr', type=float, default=3e-4, help="Learning rate for optimizer of dssm model.")
    # parse.add_argument('-dssm_emb', type=int, default=128, help="Embedding output dimension of dssm model.")
    parse.add_argument('-dssm_layers', type=int, default=3, help="Number of mlp layers for dssm model.")
    parse.add_argument('-dssm_units', type=int, default=300, help="Number of mlp units for dssm model.")
    # parameters for cdssm
    parse.add_argument('-cdssm_neg', type=int, default=4, help="Number of negative samples for ranking of cdssm model.")
    parse.add_argument('-cdssm_ngram', type=int, default=3, help="The N of n-gram model for cdssm model.")
    parse.add_argument('-cdssm_epoch', type=int, default=5, help="Number of epochs for training of cdssm model.")
    parse.add_argument('-cdssm_dropout', type=float, default=0.5, help="Dropout rate for training of cdssm model.")
    parse.add_argument('-cdssm_lr', type=float, default=3e-4, help="Learning rate for optimizer of cdssm model.")
    # parse.add_argument('-cdssm_emb', type=int, default=128, help="Embedding output dimension of cdssm model.")
    parse.add_argument('-cdssm_layers', type=int, default=1, help="Number of mlp layers for cdssm model.")
    parse.add_argument('-cdssm_units', type=int, default=64, help="Number of mlp units for cdssm model.")
    # parameters for mv_lstm
    parse.add_argument('-mv_lstm_neg', type=int, default=4,
                       help="Number of negative samples for ranking of mv_lstm model.")
    parse.add_argument('-mv_lstm_epoch', type=int, default=5, help="Number of epochs for training of mv_lstm model.")
    parse.add_argument('-mv_lstm_dropout', type=float, default=0.5, help="Dropout rate for training of mv_lstm model.")
    parse.add_argument('-mv_lstm_lr', type=float, default=3e-4, help="Learning rate for optimizer of mv_lstm model.")
    # parameters for drmm
    parse.add_argument('-drmm_neg', type=int, default=4, help="Number of negative samples for ranking of drmm model.")
    parse.add_argument('-drmm_epoch', type=int, default=5, help="Number of epochs for training of drmm model.")
    parse.add_argument('-drmm_dropout', type=float, default=0.5, help="Dropout rate for training of drmm model.")
    parse.add_argument('-drmm_lr', type=float, default=3e-4, help="Learning rate for optimizer of drmm model.")
    parse.add_argument('-drmm_emb', type=int, default=128, help="Embedding output dimension of drmm model.")
    parse.add_argument('-drmm_layers', type=int, default=1, help="Number of mlp layers for drmm model.")
    parse.add_argument('-drmm_units', type=int, default=32, help="Number of mlp units for drmm model.")
    # parameters for drmmtks
    parse.add_argument('-drmmtks_neg', type=int, default=4, help="Number of negative samples for ranking of drmmtks model.")
    parse.add_argument('-drmmtks_epoch', type=int, default=5, help="Number of epochs for training of drmmtks model.")
    parse.add_argument('-drmmtks_dropout', type=float, default=0.5, help="Dropout rate for training of drmmtks model.")
    parse.add_argument('-drmmtks_lr', type=float, default=3e-4, help="Learning rate for optimizer of drmmtks model.")
    # parameters for match_lstm
    parse.add_argument('-match_lstm_neg', type=int, default=4, help="Number of negative samples for ranking of match_lstm model.")
    parse.add_argument('-match_lstm_epoch', type=int, default=5, help="Number of epochs for training of match_lstm model.")
    parse.add_argument('-match_lstm_dropout', type=float, default=0.5, help="Dropout rate for training of match_lstm model.")
    parse.add_argument('-match_lstm_lr', type=float, default=3e-4, help="Learning rate for optimizer of match_lstm model.")
    # parameters for duet
    parse.add_argument('-duet_neg', type=int, default=4, help="Number of negative samples for ranking of duet model.")
    parse.add_argument('-duet_epoch', type=int, default=5, help="Number of epochs for training of duet model.")
    parse.add_argument('-duet_dropout', type=float, default=0.5, help="Dropout rate for training of duet model.")
    parse.add_argument('-duet_lr', type=float, default=3e-4, help="Learning rate for optimizer of duet model.")
    # parameters for knrm
    parse.add_argument('-knrm_neg', type=int, default=4, help="Number of negative samples for ranking of knrm model.")
    parse.add_argument('-knrm_epoch', type=int, default=5, help="Number of epochs for training of knrm model.")
    parse.add_argument('-knrm_dropout', type=float, default=0.5, help="Dropout rate for training of knrm model.")
    parse.add_argument('-knrm_lr', type=float, default=3e-4, help="Learning rate for optimizer of knrm model.")
    # parameters for conv_knrm
    parse.add_argument('-conv_knrm_neg', type=int, default=4, help="Number of negative samples for ranking of conv_knrm model.")
    parse.add_argument('-conv_knrm_epoch', type=int, default=5, help="Number of epochs for training of conv_knrm model.")
    parse.add_argument('-conv_knrm_dropout', type=float, default=0.5, help="Dropout rate for training of conv_knrm model.")
    parse.add_argument('-conv_knrm_lr', type=float, default=3e-4, help="Learning rate for optimizer of conv_knrm model.")
    # parameters for esim
    parse.add_argument('-esim_neg', type=int, default=4, help="Number of negative samples for ranking of esim model.")
    parse.add_argument('-esim_epoch', type=int, default=5, help="Number of epochs for training of esim model.")
    parse.add_argument('-esim_dropout', type=float, default=0.5, help="Dropout rate for training of esim model.")
    parse.add_argument('-esim_lr', type=float, default=3e-4, help="Learning rate for optimizer of esim model.")
    # parameters for bimpm
    parse.add_argument('-bimpm_neg', type=int, default=4, help="Number of negative samples for ranking of bimpm model.")
    parse.add_argument('-bimpm_epoch', type=int, default=5, help="Number of epochs for training of bimpm model.")
    parse.add_argument('-bimpm_dropout', type=float, default=0.5, help="Dropout rate for training of bimpm model.")
    parse.add_argument('-bimpm_lr', type=float, default=3e-4, help="Learning rate for optimizer of bimpm model.")
    # parameters for match_pyramid
    parse.add_argument('-match_pyramid_neg', type=int, default=4, help="Number of negative samples for ranking of match_pyramid model.")
    parse.add_argument('-match_pyramid_epoch', type=int, default=5, help="Number of epochs for training of match_pyramid model.")
    parse.add_argument('-match_pyramid_dropout', type=float, default=0.5, help="Dropout rate for training of match_pyramid model.")
    parse.add_argument('-match_pyramid_lr', type=float, default=3e-4, help="Learning rate for optimizer of match_pyramid model.")
    # parameters for match_srnn
    parse.add_argument('-match_srnn_neg', type=int, default=4, help="Number of negative samples for ranking of match_srnn model.")
    parse.add_argument('-match_srnn_epoch', type=int, default=5, help="Number of epochs for training of match_srnn model.")
    parse.add_argument('-match_srnn_dropout', type=float, default=0.5, help="Dropout rate for training of match_srnn model.")
    parse.add_argument('-match_srnn_lr', type=float, default=3e-4, help="Learning rate for optimizer of match_srnn model.")
    # parameters for anmm
    parse.add_argument('-anmm_neg', type=int, default=4, help="Number of negative samples for ranking of anmm model.")
    parse.add_argument('-anmm_epoch', type=int, default=5, help="Number of epochs for training of anmm model.")
    parse.add_argument('-anmm_dropout', type=float, default=0.5, help="Dropout rate for training of anmm model.")
    parse.add_argument('-anmm_lr', type=float, default=3e-4, help="Learning rate for optimizer of anmm model.")
    # parameters for diin
    parse.add_argument('-diin_neg', type=int, default=4, help="Number of negative samples for ranking of diin model.")
    parse.add_argument('-diin_epoch', type=int, default=5, help="Number of epochs for training of diin model.")
    parse.add_argument('-diin_dropout', type=float, default=0.5, help="Dropout rate for training of diin model.")
    parse.add_argument('-diin_lr', type=float, default=3e-4, help="Learning rate for optimizer of diin model.")
    # parameters for hbmp
    parse.add_argument('-hbmp_neg', type=int, default=4, help="Number of negative samples for ranking of hbmp model.")
    parse.add_argument('-hbmp_epoch', type=int, default=5, help="Number of epochs for training of hbmp model.")
    parse.add_argument('-hbmp_dropout', type=float, default=0.5, help="Dropout rate for training of hbmp model.")
    parse.add_argument('-hbmp_lr', type=float, default=3e-4, help="Learning rate for optimizer of hbmp model.")

    argv = parse.parse_args()
    main(argv)
