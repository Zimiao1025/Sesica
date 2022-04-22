import joblib

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer


def dataset_split(index_list):
    # constant seed
    np.random.seed(1025)
    # split node of graph a for train
    sp_num = len(index_list) // 5
    np.random.shuffle(index_list)

    return index_list[:sp_num * 3], index_list[sp_num * 3:sp_num * 4], index_list[sp_num * 4:]


def pre_fit(train_x, sc_method, scale_path):
    model_path = scale_path + sc_method + '_scale.pkl'
    if sc_method == 'mms':
        scale = MinMaxScaler()
    elif sc_method == 'ss':
        scale = StandardScaler()
    else:
        scale = Normalizer()
    scale.fit(train_x)
    joblib.dump(scale, model_path)
    return scale.transform(train_x)


def pre_trans(valid_x, sc_method, scale_path):
    model_path = scale_path + sc_method + '_scale.pkl'
    scale = joblib.load(model_path)
    return scale.transform(valid_x)


def normalize_prob(prob):
    scale = MinMaxScaler()
    scale.fit(prob)
    new_prob = scale.transform(prob)
    return new_prob.flatten()
