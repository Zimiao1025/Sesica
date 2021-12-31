import numpy as np
import pandas as pd

from plot import plot_bin, plot_curve, plot_int


def load_prob(args, ind, params):
    plot_data = {}
    for clf in args.clf:
        file_path = args.ssc_path[clf] + 'valid_prob.csv' if not ind else args.ssc_path[clf] + 'ind_prob.csv'
        clf_prob = pd.read_csv(file_path, dtype=np.float).to_dict('list')
        top_n = params['top_n'][clf]
        for n in range(1, top_n + 1):
            plot_data[clf + '_' + str(n)] = clf_prob['top_' + str(n)]
    if args.integrate != 'none':
        int_file_path = args.int_path + 'prob.npy' if not ind else args.int_path + 'ind_prob.npy'
        plot_data[args.integrate] = list(np.load(int_file_path))

    return plot_data


def load_results(args, ind, plot_metric, params):
    results = {}
    for clf in args.clf:
        top_n = params['top_n'][clf]
        for n in range(1, top_n + 1):
            file_name = 'top_' + str(n) + '_eval_results.csv' if not ind else 'top_' + str(n) + '_eval_results_ind.csv'
            eval_data = pd.read_csv(args.ssc_path[clf] + file_name, dtype=np.float).to_dict('list')[plot_metric]
            results[clf + '_' + str(n)] = eval_data
    if args.integrate != 'none':
        int_file_path = args.int_path + 'eval_results.csv' if not ind else args.int_path + 'eval_results_ind.csv'
        results[args.integrate] = pd.read_csv(int_file_path, dtype=np.float).to_dict('list')[plot_metric]

    return results


def load_metric(args, ind, plot_metric, params):
    results = {}
    for clf in args.clf:
        top_n = params['top_n'][clf]
        for n in range(1, top_n + 1):
            file_name = 'top_' + str(n) + '_eval_results.csv' if not ind else 'top_' + str(n) + '_eval_results_ind.csv'
            eval_data = pd.read_csv(args.ssc_path[clf] + file_name, dtype=np.float).to_dict('list')[plot_metric]
            results[clf + '_' + str(n)] = np.mean(eval_data)
    if args.integrate != 'none':
        int_file_path = args.int_path + 'eval_results.csv' if not ind else args.int_path + 'eval_results_ind.csv'
        results[args.integrate] = np.mean(pd.read_csv(int_file_path, dtype=np.float).to_dict('list')[plot_metric])

    return results


def check_args4plot(args):
    if 'box' in args.plot:
        args.plot_metric = args.metrics[0] if args.plot_metric == 'metric_1' else args.plot_metric
    return args


def plot_fig(args, ind, params):
    args = check_args4plot(args)
    # First step: load data
    # prc, roc, box, hp, 3d, dist, pie, bar
    print(args.plot)
    for pl in args.plot:
        if pl in ['prc', 'roc']:
            metric = 'aupr' if pl == 'prc' else 'auc'
            au_dict = load_metric(args, ind, metric, params)
            print(au_dict)
            exit()
            true_y = np.load(args.data_dir + 'valid_y.npy') if not ind else np.load(args.data_dir + 'ind_y.npy')
            prob_dict = load_prob(args, ind, params)
            if pl == 'prc':
                fig_path = args.fig_dir + 'prc.png' if not ind else args.fig_dir + 'ind_prc.png'
                plot_curve.plot_prc(true_y, prob_dict, au_dict, fig_path)
            else:
                fig_path = args.fig_dir + 'roc.png' if not ind else args.fig_dir + 'ind_roc.png'
                plot_curve.plot_roc(true_y, prob_dict, au_dict, fig_path)
        elif pl == 'box':
            result_dict = load_results(args, ind, args.plot_metric, params)
            fig_path = args.fig_dir + 'box.png' if not ind else args.fig_dir + 'ind_box.png'
            plot_bin.box_fig(pd.DataFrame(result_dict), fig_path)
        elif pl == '3d':
            args.integrate = 'none'
            prob_dict = load_prob(args, ind, params)
            true_y = np.load(args.data_dir + 'valid_y.npy') if not ind else np.load(args.data_dir + 'ind_y.npy')
            fig_path = args.fig_dir + 'dr_3d.png' if not ind else args.fig_dir + 'dr_3d_ind.png'
            plot_bin.plot_3d(np.array(list(prob_dict.values()), dtype=np.float).transpose(), true_y, fig_path)
        elif pl == 'dist':
            prob_dict = load_prob(args, ind, params)
            fig_path = args.fig_dir + 'dist.png' if not ind else args.fig_dir + 'dist_ind.png'
            plot_bin.dist_fig(pd.DataFrame(prob_dict['svm_1']), fig_path)
        elif pl == 'hp':
            args.integrate = 'none'
            prob_dict = load_prob(args, ind, params)
            fig_path = args.fig_dir + 'hp.png' if not ind else args.fig_dir + 'hp_ind.png'
            plot_bin.hp_fig(pd.DataFrame(prob_dict), fig_path)
        elif pl == 'pie':
            if args.integrate in ['ga', 'de']:
                fig_path = args.integrate + '_pie.png' if not ind else args.integrate + '_pie_ind.png'
                weight = np.load(args.int_path + 'opt_weight.npy')
                plot_int.pie_fig(weight, fig_path)
            else:
                print('If you want to plot pie fig, the integrate method ga or de should be selected!\n')
        elif pl == 'bar':
            if args.integrate == 'ltr':
                args.integrate = 'none'
                prob_dict = load_prob(args, ind, params)
                fig_path = args.integrate + '_bar.png' if not ind else args.integrate + '_bar_ind.png'
                plot_int.bar_fig(pd.DataFrame(prob_dict), args.int_path + 'ltr_model.pkl', list(prob_dict.keys()),
                                 fig_path)
            else:
                print('If you want to plot bar fig, the integrate method ltr should be selected!\n')
