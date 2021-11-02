import numpy as np


def load_data4rank(feats_file, group_file):
    """ 加载排序所需数据 """
    x_train, y_train = read_svm(feats_file)
    q_train = np.loadtxt(group_file)

    # print(x_train)
    # print(y_train)
    # print(q_train)
    return x_train, y_train, q_train


def read_svm(in_file):
    """transform svm format file to vector arrays"""
    vectors = []
    labels = []
    with open(in_file, 'r') as svmFile:
        lines = svmFile.readlines()
        for line in lines:
            vector = []
            temp_str = line.split()
            for i in range(1, len(temp_str)):
                temp_val = temp_str[i].split(':')[1]
                vector.append(float(temp_val))

            vectors.append(vector)
            labels.append(int(temp_str[0]))
    return np.array(vectors, dtype=np.float32), np.array(labels, dtype=np.float32)
