import os

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def ro_curve(true_labels, pred_prob, file_path):
    fpr_ind, tpr_ind, thresholds_ind = roc_curve(true_labels, pred_prob)
    try:
        ind_auc = auc(fpr_ind, tpr_ind)
    except ZeroDivisionError:
        ind_auc = 0.0
    plt.figure(0)
    plt.plot(fpr_ind, tpr_ind, lw=2, alpha=0.7, color='red',
             label='ROC curve (area = %0.2f)' % ind_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)

    figure_name = file_path + 'ind_roc.png'
    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
    full_path = os.path.abspath(figure_name)
    if os.path.isfile(full_path):
        print('The Receiver Operating Characteristic of independent test can be found:')
        print(full_path)
        print('\n')
    return ind_auc
