import csv
import random

import numpy as np


def gen_dataset(sample_num, edge_up_bound, fea_len, fea_wid, csv_path):
    """ 生成数据集/确定输入数据集的格式 """
    benchmark_qk = []
    index = 0
    for i in range(sample_num):
        query_name = 'q_' + str(i)
        query_fea = np.random.randn(fea_len, fea_wid)

        # 每个query 配小于 edge_up_bound 个 key, 同一个 query 所属 group 相同
        connect_num = random.randint(1, edge_up_bound)
        for j in range(connect_num):
            index += 1
            key_name = 'k_' + str(j)
            key_fea = np.random.randn(fea_len, fea_wid)
            label = 1 if random.randint(0, 9) >= 5 else 0
            one_qk = {"index": index, "label": label, "query_name": query_name, "key_name": key_name}
            for m in range(fea_len):
                for n in range(fea_wid):
                    q_tag = "qf_" + str(m) + "_" + str(n)
                    one_qk[q_tag] = round(query_fea[m][n], 5)
            for m in range(fea_len):
                for n in range(fea_wid):
                    k_tag = "kf_" + str(m) + "_" + str(n)
                    one_qk[k_tag] = round(key_fea[m][n], 5)
            benchmark_qk.append(one_qk)
    # print(benchmark_qk[0])
    save_csv(benchmark_qk, csv_path)


def save_csv(qk_dict_list, file_name):

    with open(file_name, "w", newline="") as csvFile:

        # 文件头以列表的形式传入函数，列表的每个元素表示每一列的标识
        file_header = list(qk_dict_list[0].keys())
        dict_writer = csv.DictWriter(csvFile, file_header)
        dict_writer.writeheader()

        # 之后，按照（属性：数据）的形式，将字典写入CSV文档即可
        for qk_dict in qk_dict_list:
            dict_writer.writerow(qk_dict)


def read_csv(file_name):
    qk_dict_list = []
    with open(file_name, "r") as csvFile:
        dict_reader = csv.DictReader(csvFile)
        # 输出第一行，也就是数据名称那一行
        # print(dict_reader.fieldnames)
        for row in dict_reader:
            qk_dict_list.append(row)
    return qk_dict_list


def dataset_split(index_list):
    # constant seed
    np.random.seed(1025)
    # split node of graph a for train
    sp_num = len(index_list) // 5
    index_list = np.array(index_list)
    np.random.shuffle(index_list)

    return index_list[sp_num:], index_list[:sp_num]
