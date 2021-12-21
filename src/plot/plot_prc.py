import os

import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve


def pr_curve(true_labels, pred_prob, file_path):
    precision, recall, _ = precision_recall_curve(true_labels, pred_prob)
    try:
        ind_auc = auc(recall, precision)
    except ZeroDivisionError:
        ind_auc = 0.0
    plt.figure(0)
    plt.plot(recall, precision, lw=2, alpha=0.7, color='red',
             label='PRC curve (area = %0.2f)' % ind_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Precision-Recall Curve', fontsize=18)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.legend(loc="lower left")
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)

    figure_name = file_path + 'ind_prc.png'
    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(figure_name)
    if os.path.isfile(full_path):
        print('The Precision-Recall Curve of independent test can be found:')
        print(full_path)
        print('\n')
    return ind_auc
