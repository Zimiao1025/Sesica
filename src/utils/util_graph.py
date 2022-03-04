from collections import defaultdict

import numpy as np


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
