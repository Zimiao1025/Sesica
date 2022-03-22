from rank.rank_process import int_or_rank
from utils import util_ctrl


def main(args):
    print("\n******************************** RANK ********************************\n")
    args = util_ctrl.rank_path_ctrl(args)
    params = util_ctrl.params_rank(args)
    int_or_rank(args, params)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='Sesica', description="Step into analysis, please select parameters ")

    # parameters for rank
    parse.add_argument('-base_dir', nargs='*', required=True, help="The path to store result.")
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
    # parameters for integration
    parse.add_argument('-rank', type=str, choices=['ltr', 'none'], default='none',
                       help="Rank by:\n"
                            " 'none' --- Without integration, the output is sorted directly according to the metric;\n"
                            " 'ltr' --- Learning to rank with LambdaRank.\n"
                       )
    # parameters for ltr
    parse.add_argument('-ltr_m', type=int, default=[0], nargs='*',
                       help="Maximum tree depth for base learners, <=0 means no limit.")
    parse.add_argument('-ltr_t', type=int, default=[100], nargs='*', help="Number of boosted trees to fit.")
    parse.add_argument('-ltr_n', type=int, default=[31], nargs='*', help="Maximum tree leaves for base learners.")
    argv = parse.parse_args()
    main(argv)
