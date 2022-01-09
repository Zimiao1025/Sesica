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
        if ind or args.integrate not in ['lr', 'ltr']:
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
        if ind or args.integrate not in ['lr', 'ltr']:
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
        if ind or args.integrate not in ['lr', 'ltr']:
            int_file_path = args.int_path + 'eval_results.csv' if not ind else args.int_path + 'eval_results_ind.csv'
            results[args.integrate] = np.mean(pd.read_csv(int_file_path, dtype=np.float).to_dict('list')[plot_metric])

    return results


def load_eval(args, polar_clf, ind, params):
    results = {}
    for clf in polar_clf:
        top_n = params['top_n'][clf]
        for n in range(1, top_n + 1):
            file_name = 'top_' + str(n) + '_eval_results.csv' if not ind else 'top_' + str(n) + '_eval_results_ind.csv'
            eval_data = pd.read_csv(args.ssc_path[clf] + file_name, dtype=np.float).mean().tolist()
            results[clf + '_' + str(n)] = eval_data[1:]
    if args.integrate != 'none':
        if ind or args.integrate not in ['lr', 'ltr']:
            int_file_path = args.int_path + 'eval_results.csv' if not ind else args.int_path + 'eval_results_ind.csv'
            int_eval_data = pd.read_csv(int_file_path, dtype=np.float).mean().tolist()
            results[args.integrate] = int_eval_data[1:]

    return results


def data_rus(vec_arr, prob_arr, label_arr):
    from collections import Counter
    from imblearn.under_sampling import RandomUnderSampler
    print('Original dataset shape %s' % Counter(label_arr))

    rus = RandomUnderSampler(random_state=1025)
    all_fea = np.hstack((vec_arr, prob_arr))
    X_res, y_res = rus.fit_resample(all_fea, label_arr)
    print('Resampled dataset shape %s' % Counter(y_res))
    dim = vec_arr.shape[1]
    return X_res[:, :dim], X_res[:, dim:], y_res


def prob_rus(prob_dict, label_arr, dist_method):
    from collections import Counter
    from imblearn.under_sampling import RandomUnderSampler
    print('Original dataset shape %s' % Counter(label_arr))
    prob_list = []
    for method in dist_method:
        prob_list.append(prob_dict[method])
    rus = RandomUnderSampler(random_state=1025)
    prob_arr = np.array(prob_list).transpose()
    print(prob_arr.shape)
    X_res, y_res = rus.fit_resample(prob_arr, label_arr)
    print(X_res[0])
    print('Resampled dataset shape %s' % Counter(y_res))
    return X_res


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
            print('au_dict: ', au_dict)
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
        elif pl == 'polar':
            polar_method = ['rsvm', 'lsvm']
            result_dict = load_eval(args, polar_method, ind, params)
            fig_path = args.fig_dir + 'polar.png' if not ind else args.fig_dir + 'ind_polar.png'
            plot_bin.polar_fig(polar_method, list(result_dict.values()), args.metrics, fig_path)
        elif pl == '3d':
            args.integrate = 'none'
            prob_dict = load_prob(args, ind, params)
            prob_arr = np.array(list(prob_dict.values()), dtype=np.float).transpose()
            true_x = np.load(args.data_dir + 'valid_x.npy') if not ind else np.load(args.data_dir + 'ind_x.npy')
            true_y = np.load(args.data_dir + 'valid_y.npy') if not ind else np.load(args.data_dir + 'ind_y.npy')
            new_x, new_prob, new_y = data_rus(true_x, prob_arr, true_y)
            old_fig_path = args.fig_dir + 'old_dr_3d.png' if not ind else args.fig_dir + 'old_dr_3d_ind.png'
            plot_bin.plot_3d(new_x, new_y, old_fig_path)
            new_fig_path = args.fig_dir + 'new_dr_3d.png' if not ind else args.fig_dir + 'new_dr_3d_ind.png'
            plot_bin.plot_3d(new_prob, new_y, new_fig_path)
        elif pl == 'dist':
            dist_method = ['rsvm_1', 'lsvm_1']
            true_y = np.load(args.data_dir + 'valid_y.npy') if not ind else np.load(args.data_dir + 'ind_y.npy')
            prob_dict = load_prob(args, ind, params)
            new_prob = prob_rus(prob_dict, true_y, dist_method)
            fig_path = args.fig_dir + 'dist.png' if not ind else args.fig_dir + 'dist_ind.png'
            plot_bin.dist_fig(new_prob, dist_method, fig_path)
        elif pl == 'hp':
            args.integrate = 'none'
            prob_dict = load_prob(args, ind, params)
            fig_path = args.fig_dir + 'hp.png' if not ind else args.fig_dir + 'hp_ind.png'
            plot_bin.hp_fig(pd.DataFrame(prob_dict), fig_path)
        elif pl == 'pie':
            if args.integrate in ['ga', 'de']:
                fig_path = args.fig_dir + '_pie.png' if not ind else args.fig_dir + '_pie_ind.png'
                weight = np.load(args.int_path + 'opt_weight.npy')
                plot_int.pie_fig(weight, args.clf, fig_path)
            else:
                print('If you want to plot pie fig, the integrate method ga or de should be selected!\n')
        elif pl == 'bar':
            if args.integrate == 'ltr':
                args.integrate = 'none'
                prob_dict = load_prob(args, ind, params)
                fig_path = args.fig_dir + '_bar.png' if not ind else args.fig_dir + '_bar_ind.png'
                plot_int.bar_fig(args.int_path + 'ltr_model.pkl', list(prob_dict.keys()), fig_path)
            else:
                print('If you want to plot bar fig, the integrate method ltr should be selected!\n')