import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.stats import pearsonr

SEED = 42


# 欧氏距离
def euclidean_distance(vec1, vec2):
    # ord=2: 二范数
    score = np.linalg.norm(vec1-vec2, ord=2)
    return round(score, 4)


# 曼哈顿距离
def manhattan_distance(vec1, vec2):
    # ord=1: 一范数
    score = np.linalg.norm(vec1 - vec2, ord=1)
    return round(score, 4)


# 切比雪夫距离
def chebyshev_distance(vec1, vec2):
    # ord=np.inf: 无穷范数
    score = np.linalg.norm(vec1-vec2, ord=np.inf)
    return round(score, 4)


# 汉明距离
def hamming_distance(vec1, vec2):
    # 适用于二进制编码格式 !
    return len(np.nonzero(vec1-vec2)[0])  # 返回整数


# 杰卡德相似度
def jaccard_similarity_coefficient(vec1, vec2):
    # 适用于二进制编码格式 !
    score = distance.pdist(np.array([vec1, vec2]), "jaccard")[0]
    return round(score, 4)


# 余弦相似度
def cosine_similarity(vec1, vec2):
    score = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*(np.linalg.norm(vec2)))
    return round(score, 4)


# 皮尔森相关系数
def pearson_correlation_coefficient(vec1, vec2):
    score = pearsonr(vec1, vec2)[0]
    return round(score, 4)


# 相对熵又称交叉熵, Kullback-Leible散度（即KL散度）等,
# 这个指标不能用作距离衡量，因为该指标不具有对称性, 为了在并行计算时统一，采用对称KL散度
def kl_divergence(vec1, vec2):
    score = (entropy(vec1, vec2) + entropy(vec2, vec1)) / 2.0
    return round(score, 4)


def score_func(method, query, key):
    """ 根据选择的函数进行无监督打分 """
    query = query.flatten()
    key = key.flatten()
    print(query)
    print(key)
    if method == 'ED':
        score = euclidean_distance(query, key)
    elif method == 'MD':
        score = manhattan_distance(query, key)
    elif method == 'CD':
        score = chebyshev_distance(query, key)
    elif method == 'HD':
        score = hamming_distance(query, key)
    elif method == 'JSC':
        score = jaccard_similarity_coefficient(query, key)
    elif method == 'CS':
        score = cosine_similarity(query, key)
    elif method == 'PCC':
        score = pearson_correlation_coefficient(query, key)
    elif method == 'KLD':
        score = kl_divergence(query, key)
    else:
        print('Semantic Similarity method error!')
        return False
    return score
