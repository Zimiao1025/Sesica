import joblib
import matplotlib.pyplot as plt
import numpy as np


def bar_fig(model_path, feats_name, fig_path):
    # 利用shap打印特征重要度
    gbm = joblib.load(model_path)
    scores = gbm.feature_importances_
    print(scores)
    plt.figure(0)
    bar_width = 0.35
    x1 = []
    for i in range(len(scores)):
        x1.append(i)
    plt.bar(x1, scores, bar_width, color='crimson', align="center",  label="scores", alpha=0.5)
    plt.xticks(x1, feats_name, size=10)
    plt.title('Feature importance for LTR', fontsize=18)
    plt.xlabel('method', fontsize=16)
    plt.ylabel('Feature importance', fontsize=16)
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)
    plt.savefig(fig_path, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)


def pie_fig(data, recipe, fig_path):
    print(recipe)
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    # startangle 设置方向
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    # 每一类别说明框
    # boxstyle框的类型，fc填充颜色,ec边框颜色,lw边框宽度
    bbox_props = dict(boxstyle="square,pad=0.3", fc='white', ec="black", lw=0.72)
    # 设置框引出方式
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    # 添加标签
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        # 设置方向
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        # 设置标注
        ax.annotate(recipe[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, color='black', **kw)

    # ax.set_title("Matplotlib bakery: A donut")

    plt.savefig(fig_path)
