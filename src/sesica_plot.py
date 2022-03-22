from plot import plot_process
from utils import util_ctrl


def main(args):
    print("\n******************************** PLOT ********************************\n")
    args = util_ctrl.plot_path_ctrl(args)
    params = util_ctrl.params_plot(args)
    plot_process.plot_fig(args, False, params)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='Sesica', description="Step into analysis, please select parameters ")

    # parameters for plot
    parse.add_argument('-base_dir', required=True, help="The relative path or absolute path to store result.")
    parse.add_argument('-ind', choices=[True, False], default=False,
                       help="The input files for positive and negative associations.")
    parse.add_argument('-clf', type=str, nargs='*',
                       choices=['svm', 'rf', 'ert', 'knn', 'mnb', 'gbdt', 'dart', 'goss', 'mlp', 'none'],
                       default='none')
    parse.add_argument('-arc', type=str, nargs='*',
                       choices=['arci', 'arcii', 'dssm', 'cdssm', 'drmm', 'drmmtks', 'match_lstm', 'duet', 'knrm',
                                'conv_knrm', 'esim', 'bimpm', 'match_pyramid', 'match_srnn', 'anmm', 'mv_lstm', 'diin',
                                'hbmp', 'none'],
                       default='none')
    parse.add_argument('-rank', type=str, choices=['ltr', 'none'], default='none',
                       help="Rank by:\n"
                            " 'none' --- Without integration, the output is sorted directly according to the metric;\n"
                            " 'ltr' --- Learning to rank with LambdaRank.\n"
                       )
    parse.add_argument('-plot', type=str, choices=['prc', 'roc', 'box', 'polar', 'hp', '3d', 'dist', 'pie', 'bar',
                                                   'none'], default='none', nargs='*',
                       help="Plot:\n"
                            " 'none' --- Don't plot;\n"
                            " 'prc' --- precision-recall Curve; 'roc' --- receiver operating characteristic;\n"
                            " 'box' --- box figure for evaluation results; 'hp' --- heat map of the relevance.\n"
                            " '3d' --- 3d figure for dimension reduce; 'dist' --- histogram for distribution.\n"
                            " 'pie' --- pie figure for optimal weight; 'bar' --- histogram for feature importance.\n"
                       )
    parse.add_argument('-top_n', type=int, nargs='*', default=[1],
                       help="Select the n best models for specific metric of distribution based methods.")
    parse.add_argument('-plot_metric', type=str, choices=['aupr', 'auc', 'ndcg@k', 'roc@k', 'metric_1'],
                       default='metric_1', help="The metrics for plot, the -plot_metric should be a metric included in "
                                                "-metric parameter you chose before. The metric_1 means the first "
                                                "metric you chose in -metrics parameter")
    argv = parse.parse_args()
    main(argv)
