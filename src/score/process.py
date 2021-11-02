from collections import Counter
import numpy as np

from function import score_func


def preprocess4score(qk_dict_list):
    labels = []
    query_names = []
    query_feats = []
    key_feats = []
    for qk_dict in qk_dict_list:
        labels.append(qk_dict["label"])
        query_names.append(qk_dict["query_name"])
        # 分别存储
        qf_dict = {}
        kf_dict = {}
        for key in qk_dict.keys():
            if "qf" in key:
                qf_dict[key] = float(qk_dict[key])
            if "kf" in key:
                kf_dict[key] = float(qk_dict[key])
        query_feats.append(qk_dict2arr(qf_dict))
        key_feats.append(qk_dict2arr(kf_dict))

    print(labels)
    print(query_names)
    print(query_feats)
    return labels, query_names, query_feats, key_feats


def qk_dict2arr(feats_dict):
    print(feats_dict)
    max_key = list(feats_dict.keys())[-1]
    print(max_key)
    row = int(max_key.split('_')[1])
    col = int(max_key.split('_')[2])
    # print(row)
    # print(col)
    return np.array(list(feats_dict.values())).reshape([row+1, col+1])


def score_process(qk_dict_list, score_methods, feats_path, group_path):
    """ 根据选择的打分函数进行相似性计算 """
    scores_list = []
    labels, query_names, query_feats, key_feats = preprocess4score(qk_dict_list)
    for query, key in zip(query_feats, key_feats):
        scores = []
        for score_method in score_methods:
            score = score_func(score_method, query, key)
            scores.append(score)
        scores_list.append(scores)

    # 写入 feats.txt
    write_feats(scores_list, labels, feats_path)
    # 写入 group.txt
    group = queries2group(query_names)
    write_group(group, group_path)


def queries2group(query_names):
    """ 输入每个qk对的query_name, 返回group """
    queries_counter = Counter(query_names)
    query_group = {}
    for query_name in query_names:
        if query_name not in query_group.keys():
            query_group[query_name] = queries_counter[query_name]

    return list(query_group.values())


def write_feats(vectors, labels, out_name):
    """以 libSVM format 写入文件，为输入ltr做准备."""
    with open(out_name, 'w') as f:
        for ind_row, label in enumerate(labels):
            temp_write = str(label)
            for ind_col, val in enumerate(vectors[ind_row]):
                temp_write += ' ' + str(ind_col + 1) + ':' + str(vectors[ind_row][ind_col])
            f.write(temp_write)
            f.write('\n')


def write_group(group, group_file):
    with open(group_file, 'w') as f:
        for g in group:
            f.write(str(g))
            f.write('\n')
