from itertools import cycle

import joblib
import matplotlib.pyplot as plt
import numpy as np


def bar_fig(model_path, feats_name, fig_path):
    # 利用shap打印特征重要度
    gbm = joblib.load(model_path)
    scores = gbm.feature_importances_
    print(scores)
    plt.figure(0)
    bar_width = 0.4
    x1 = []
    for i in range(len(scores)):
        x1.append(i)
    plt.bar(x1, scores, bar_width, color='crimson', align="center",  label="scores", alpha=0.8)
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
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(0)


def pie_fig(model_path, recipe, fig_path):
    gbm = joblib.load(model_path)
    scores = gbm.feature_importances_
    color_sets = cycle(['crimson', 'navy', 'teal', 'darkorange', 'purple', 'gray', 'green', 'dodgerblue', 'gold',
                        'lightcoral', 'red'])
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            # 同时显示数值和占比的饼图
            return '{p:.1f}% ({v:d})'.format(p=pct,v=val)
        return my_autopct
    # weight = scores / sum(scores)
    # print(weight)
    _, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
    # startangle 设置方向
    # wedges, texts = ax.pie(scores, wedgeprops=dict(width=0.5), startangle=-40)
    # x_0 = [1,0,0,0] 
    # ax.pie(x_0, radius=0.9,colors = 'w')
    ax.pie(scores,startangle = 90,radius=1.3,pctdistance = 0.9,colors=color_sets,autopct=make_autopct(scores),textprops = {'fontsize':13, 'color':'k'})
    
    ax.legend(recipe, bbox_to_anchor=(1.0,1.0), loc='center left')
    
    # 每一类别说明框
    # boxstyle框的类型，fc填充颜色,ec边框颜色,lw边框宽度
    # bbox_props = dict(boxstyle="square,pad=0.3", fc='white', ec="black", lw=0.72)
    # # 设置框引出方式
    # kw = dict(arrowprops=dict(arrowstyle="-"),
    #           bbox=bbox_props, zorder=0, va="center")

    # # 添加标签
    # for i, p in enumerate(wedges):
    #     ang = (p.theta2 - p.theta1) / 2. + p.theta1
    #     y = np.sin(np.deg2rad(ang))
    #     x = np.cos(np.deg2rad(ang))
    #     # 设置方向
    #     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    #     connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    #     kw["arrowprops"].update({"connectionstyle": connectionstyle})
    #     # 设置标注
    #     ax.annotate(recipe[i], xy=(x, y), xytext=(1.1 * np.sign(x), 1.2 * y),
    #                 horizontalalignment=horizontalalignment, color='black', **kw)
    ax.text(0.1, 1.6, 'Feature importance for LTR', fontsize=18, ha='center',va='top',wrap=True)
    # plt.title('Feature importance for LTR', fontsize=18)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(0)
