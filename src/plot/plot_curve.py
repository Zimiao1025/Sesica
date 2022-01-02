import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


def plot_roc(true_y, prob_dict, auc_dict, fig_path):
    plt.figure(0)
    count = 0
    color_list = ['crimson', 'navy', 'teal', 'darkorange', 'purple', 'gray',
                  'green', 'dodgerblue', 'gold', 'lightcoral', 'red']
    for key in prob_dict.keys():
        fpr, tpr, thresholds = roc_curve(true_y, prob_dict[key])
        annotation = key + ' (AUC = %0.3f)' % auc_dict[key]
        plt.plot(fpr, tpr, lw=2, alpha=0.7, color=color_list[count],
                 label=annotation)
        count += 1
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

    plt.savefig(fig_path, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)


def plot_prc(true_y, prob_dict, aupr_dict, fig_path):
    plt.figure(0)
    count = 0
    color_list = ['crimson', 'navy', 'teal', 'darkorange', 'purple', 'gray',
                  'green', 'dodgerblue', 'gold', 'lightcoral', 'red']
    for key in prob_dict.keys():
        precision, recall, _ = precision_recall_curve(true_y, prob_dict[key])
        annotation = key + ' (AUPR = %0.3f)' % aupr_dict[key]
        plt.plot(recall, precision, lw=2, alpha=0.7, color=color_list[count],
                 label=annotation)
        count += 1
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

    plt.savefig(fig_path, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)
