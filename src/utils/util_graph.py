from collections import defaultdict

import numpy as np
import pandas as pd


def load_vec_encodings(vec_file):
    with open(vec_file, 'r') as f:
        lines = f.readlines()
        vec_dict_list = []
        vectors = []
        for idx, line in enumerate(lines):
            # vec_id = idx // 2
            if line.startswith('>'):
                vec_name = line.strip().split()[1]
                vector = lines[idx+1].strip().split()
                vector = list(map(float, vector))
                vectors.append(vector)
                tmp_dict = {'vec_name': vec_name, 'vec_value': vector}
                vec_dict_list.append(tmp_dict)

    return pd.DataFrame(vec_dict_list), np.array(vectors, dtype=float)


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
