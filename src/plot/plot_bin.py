import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE


def plot_3d(data, labels, fig_path):
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
    axes3d = Axes3D(fig)

    for i in range(len(data_3d)):
        axes3d.scatter(data_3d[i][0], data_3d[i][1], data_3d[i][2], s=40, c=mc[i][1], alpha=0.7)

    plt.title('3D-figure of dimension reduction', fontsize=18)
    plt.xlabel('First PC', fontsize=12)
    plt.ylabel('Second PC', fontsize=12)
    axes3d.set_zlabel('Third PC', fontsize=12)

    plt.savefig(fig_path, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(fig_path)
    if os.path.isfile(full_path):
        print('The output 3D-figure for dimension reduction can be found:')
        print(full_path)
        print('\n')


def box_fig(dt, fig_path):
    plt.boxplot(x=dt.values, labels=dt.columns, whis=1.5)  # columns列索引，values所有数值
    plt.savefig(fig_path)
    plt.close(0)
    # sns.boxplot(x="distance", y="method", data=dt, whis=[0, 100], width=.6, palette="vlag")


def dist_fig(dt, fig_path):
    sns.set_palette("hls")  # 设置所有图的颜色，使用hls色彩空间
    sns.displot(x=dt.values, color="r", bins=30, kde=True)
    plt.savefig(fig_path)
    plt.close(0)


def hp_fig(dt, fig_path):
    sns.set_theme(style="white")

    # Compute the correlation matrix
    corr = dt.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    c_map = sns.diverging_palette(250, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=c_map, vmax=.3, center=0, mask=mask,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    f.savefig(fig_path)  # 减少边缘空白
