import numpy as np


def dataset_split(index_list):
    # constant seed
    np.random.seed(1025)
    # split node of graph a for train
    sp_num = len(index_list) // 5
    np.random.shuffle(index_list)

    return index_list[:sp_num*3], index_list[sp_num*3:sp_num*4], index_list[sp_num*4:]
