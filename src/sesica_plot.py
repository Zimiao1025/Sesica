from plot import plot_process
from utils import util_ctrl


def main(args):
    print("\n******************************** PLOT ********************************\n")
    args = util_ctrl.plot_path_ctrl(args)
    params = util_ctrl.params_plot(args)
    plot_process.plot_fig(args, args.ind, args.plot_set, params)


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
                            " 'ltr' --- Learning to rank with LambdaRank.\n")
    parse.add_argument('-plot_set', type=str, choices=['valid', 'test'], default='valid',
                       help="Plot the results on validation dataset or testing dataset.\n")
    parse.add_argument('-plot', type=str, choices=['prc', 'roc', 'box', 'polar', 'hp', 'dr', 'dist', 'pie', 'bar',
                                                   'none'], default='none', nargs='*',
                       help="Plot:\n"
                            " 'none' --- Don't plot;\n"
                            " 'prc' --- precision-recall Curve; 'roc' --- receiver operating characteristic;\n"
                            " 'box' --- box figure for evaluation results; 'hp' --- heat map of the relevance.\n"
                            " 'dr' --- 3d figure for dimension reduce; 'dist' --- histogram for distribution.\n"
                            " 'pie' --- pie figure for feature importance; 'bar' --- histogram for feature importance.\n")
    parse.add_argument('-top_n', type=int, nargs='*', default=[1],
                       help="Select the n best models for specific metric of distribution based methods.")
    parse.add_argument('-metric', type=str,
                       choices=['aupr', 'auc', 'ndcg', 'roc@1', 'ndcg@10', 'roc@10', 'ndcg@20', 'roc@20', 'ndcg@50',
                                'roc@50'], default='aupr', help="The metrics used for parameters selection")
    argv = parse.parse_args()
    main(argv)
