from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE


def polar_fig(methods, val_list, metric_list, fig_path):
    # use ggplot style
    plt.style.use('ggplot')
    N = len(val_list[1])
    # 设置雷达图的角度，用于平分切开一个平面
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # 使雷达图封闭起来
    cyc_val_list = []
    for val in val_list:
        cyc_val_list.append(np.concatenate((val, [val[0]])))

    # 绘图
    fig = plt.figure()
    # 设置为极坐标格式
    ax = fig.add_subplot(111, polar=True)
    # 绘制折线图
    color_sets = cycle(['crimson', 'navy', 'teal', 'darkorange', 'purple', 'gray', 'green', 'dodgerblue', 'gold',
                        'lightcoral', 'red'])
    for method, cyc_val, cor in zip(methods, cyc_val_list, color_sets):
        ax.plot(angles, cyc_val, 'o-', linewidth=2, label=method)
        ax.fill(angles, cyc_val, cor, alpha=0.1)

    angle = np.deg2rad(0)
    ax.legend(loc="lower left",
              bbox_to_anchor=(.5 + np.cos(angle) / 2, .5 + np.sin(angle) / 2))
    # 添加每个特质的标签
    metric_list = np.concatenate((metric_list, [metric_list[0]]))
    ax.set_thetagrids(angles * 180 / np.pi, metric_list)
    # 设置极轴范围
    ax.set_ylim(0, 1)
    # 添加标题
    plt.title('Polar fig')
    # 增加网格纸
    ax.grid(True)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(0)


def plot_3d(data, labels, fig_path, old=True):
    try:
        # print(data)
        data_3d = TSNE(n_components=3, init='pca', random_state=1025).fit_transform(data)
    except RuntimeWarning:
        data_3d = KernelPCA(n_components=3, random_state=1025).fit_transform(data)

    mark_sets = cycle(['o', 'o'])
    color_sets = cycle(['crimson', 'navy', 'teal', 'darkorange', 'slategrey'])
    label_set = list(set(labels))
    my_dict = {}
    m = 0
    for i in label_set:
        my_dict[i] = m
        m = m + 1
    mark_set = []
    color_set = []
    for i, j, k in zip(label_set, mark_sets, color_sets):
        mark_set.append(j)
        color_set.append(k)
    mc = np.zeros((len(labels), 2)).astype(str)
    for i in range(len(labels)):
        mc[i][0], mc[i][1] = mark_set[my_dict[labels[i]]], color_set[my_dict[labels[i]]]
    fig = plt.figure(0)
    # ax = fig.add_subplot(111)
    axes3d = Axes3D(fig)

    for i in range(len(data_3d)):
        axes3d.scatter(data_3d[i][0], data_3d[i][1], data_3d[i][2], s=40, c=mc[i][1], alpha=0.7)

    if old:
        plt.title('3D-figure of raw feature', fontsize=16)
    else:
        plt.title('3D-figure of score feature', fontsize=16)
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1), labels=['pos'])
    plt.xlabel('First PC', fontsize=12)
    plt.ylabel('Second PC', fontsize=12)
    axes3d.set_zlabel('Third PC', fontsize=12)

    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(0)


def box_fig(aupr_list, auc_list, ndcg_list, labels, fig_path):
    # set parameters for fig
    x_location = np.linspace(2, len(labels), len(labels))  # len(data个序列)
    width = 0.2
    colors = ['crimson', 'navy', 'teal']

    fig = plt.figure(figsize=(16, 9))
    # fig.tight_layout()
    ax1 = fig.add_subplot(111)
    rect_1 = ax1.bar(x_location - width, aupr_list, width=width, color=colors[0], linewidth=1, alpha=0.7)
    rect_2 = ax1.bar(x_location, auc_list, width=width, color=colors[1], linewidth=1, alpha=0.7)
    rect_3 = ax1.bar(x_location + width, ndcg_list, width=width, color=colors[2], linewidth=1, alpha=0.7)
    # 添加x轴标签
    plt.xticks(x_location + width, labels, fontsize=12, rotation=20)  # 横坐标轴标签 rotation x轴标签旋转的角度

    # 图例
    ax1.legend((rect_1, rect_2, rect_3), (u'aupr', u'auc', u'ndcg'), fontsize=14)  # 图例
    # 添加数据标签
    for r1, r2, r3, amount01, amount02, amount03 in zip(rect_1, rect_2, rect_3, aupr_list, auc_list, ndcg_list):
        plt.text(r1.get_x(), r1.get_height(), round(amount01, 2), va='bottom')
        plt.text(r2.get_x(), r2.get_height(), round(amount02, 2), va='bottom')
        plt.text(r3.get_x(), r3.get_height(), round(amount03, 2), va='bottom')

    plt.title('The histogram for evaluation metrics', fontsize=18)
    plt.xlabel('Base method', fontsize=14, labelpad=10)
    plt.ylabel('Value', fontsize=14, labelpad=10)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(0)


def dist_fig(prob_arr, dist_method, fig_path):
    sns.set_palette("hls")  # 设置所有图的颜色，使用hls色彩空间
    color_sets = ['crimson', 'navy', 'teal', 'darkorange', 'purple', 'gray', 'green', 'dodgerblue', 'gold',
                  'lightcoral', 'red']
    # print(prob_arr)
    # print(prob_arr[:, 1])
    fig = plt.figure()
    # fig.tight_layout()
    ax = fig.add_subplot(111)
    for i in range(len(dist_method)):
        plt.hist(prob_arr[:, i], bins=20, rwidth=0.5, alpha=0.5, histtype='bar', color=color_sets[i],
                 label=dist_method[i])
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.title('Frequency distribution histogram', fontsize=18)
    plt.xlabel('Score', fontsize=14, labelpad=10)
    plt.ylabel('Frequency', fontsize=14, labelpad=10)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(0)


def hp_fig(dt, fig_path):
    sns.set_theme(style="white")

    # Compute the correlation matrix
    corr = dt.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.figure()

    cmp = sns.diverging_palette(250, 10, as_cmap=True)  # "RdBu_r"
    sns.heatmap(corr, cmap=cmp, mask=mask, annot=True,
                square=True, linewidths=.5, fmt='.2f')

    plt.title('The correlation of different methods', fontsize=18)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(0)
