import numpy as np
import pandas as pd

from plot import plot_bin, plot_curve, plot_int


def load_prob(args, ind, plot_set, params):
    plot_data = {}
    for clf in args.clf:
        if plot_set == 'valid':
            file_path = args.ssc_path[clf] + 'valid_prob.csv' if not ind else args.ssc_path[clf] + 'ind_prob.csv'
        else:
            file_path = args.ssc_path[clf] + 'test_prob.csv' if not ind else args.ssc_path[clf] + 'ind_prob.csv'
        clf_prob = pd.read_csv(file_path, dtype=np.float).to_dict('list')
        top_n = params['top_n'][clf]
        for n in range(1, top_n + 1):
            plot_data[clf + '_' + str(n)] = clf_prob['top_' + str(n)]
    if args.rank == 'ltr' and plot_set == 'test':
        int_file_path = args.int_path + 'prob.npy' if not ind else args.int_path + 'ind_prob.npy'
        plot_data[args.rank] = list(np.load(int_file_path))

    return plot_data


def load_results(args, ind, plot_set, params):
    labels = []
    aupr_list = []
    auc_list = []
    ndcg_list = []
    for clf in args.clf:
        top_n = params['top_n'][clf]
        for n in range(1, top_n + 1):
            if plot_set == 'valid':
                file_name = 'top_' + str(n) + '_eval_results.csv' if not ind else 'top_' + str(n) + '_eval_results_ind.csv'
            else:
                file_name = 'top_' + str(n) + '_test_results.csv' if not ind else 'top_' + str(n) + '_test_results_ind.csv'
            file_path = args.ssc_path[clf] + file_name
            aupr_list.append(np.mean(pd.read_csv(file_path, dtype=np.float).to_dict('list')['aupr']))
            auc_list.append(np.mean(pd.read_csv(file_path, dtype=np.float).to_dict('list')['auc']))
            ndcg_list.append(np.mean(pd.read_csv(file_path, dtype=np.float).to_dict('list')['ndcg']))
            labels.append(clf + '_' + str(n))
    if args.rank != 'ltr' and plot_set == 'test':
        rank_file_path = args.int_path + 'eval_results.csv' if not ind else args.int_path + 'eval_results_ind.csv'
        aupr_list.append(np.mean(pd.read_csv(rank_file_path, dtype=np.float).to_dict('list')['aupr']))
        auc_list.append(np.mean(pd.read_csv(rank_file_path, dtype=np.float).to_dict('list')['auc']))
        ndcg_list.append(np.mean(pd.read_csv(rank_file_path, dtype=np.float).to_dict('list')['ndcg']))
        labels.append('ltr')

    return aupr_list, auc_list, ndcg_list, labels


def load_metric(args, ind, plot_set, plot_metric, params):
    results = {}
    for clf in args.clf:
        top_n = params['top_n'][clf]
        for n in range(1, top_n + 1):
            if plot_set == 'valid':
                file_name = 'top_' + str(n) + '_eval_results.csv' if not ind else 'top_' + str(n) + '_test_results_ind.csv'
            else:
                file_name = 'top_' + str(n) + '_test_results.csv' if not ind else 'top_' + str(n) + '_test_results_ind.csv'
            eval_data = pd.read_csv(args.ssc_path[clf] + file_name, dtype=np.float).to_dict('list')[plot_metric]
            results[clf + '_' + str(n)] = np.mean(eval_data)
    if args.rank == 'ltr' and plot_set == 'test':
        int_file_path = args.int_path + 'eval_results.csv' if not ind else args.int_path + 'eval_results_ind.csv'
        results[args.rank] = np.mean(pd.read_csv(int_file_path, dtype=np.float).to_dict('list')[plot_metric])

    return results


def load_eval(args, ind, plot_set, params):
    results = {}
    for clf in args.clf:
        top_n = params['top_n'][clf]
        for n in range(1, top_n + 1):
            if plot_set == 'valid':
                file_name = 'top_' + str(n) + '_eval_results.csv' if not ind else 'top_' + str(n) + '_eval_results_ind.csv'
            else:
                file_name = 'top_' + str(n) + '_test_results.csv' if not ind else 'top_' + str(n) + '_test_results_ind.csv'
            eval_data = pd.read_csv(args.ssc_path[clf] + file_name, dtype=np.float).mean().tolist()
            results[clf + '_' + str(n)] = eval_data[1:]
    if args.rank == 'ltr' and plot_set == 'test':
        int_file_path = args.int_path + 'eval_results.csv' if not ind else args.int_path + 'eval_results_ind.csv'
        int_eval_data = pd.read_csv(int_file_path, dtype=np.float).mean().tolist()
        results[args.rank] = int_eval_data[1:]

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


def prob_rus(prob_dict, label_arr):
    from collections import Counter
    from imblearn.under_sampling import RandomUnderSampler
    print('Original dataset shape %s' % Counter(label_arr))
    prob_list = list(prob_dict.values())
    rus = RandomUnderSampler(random_state=1025)
    prob_arr = np.array(prob_list).transpose()
    # print(prob_arr.shape)
    X_res, y_res = rus.fit_resample(prob_arr, label_arr)
    # print(X_res[0])
    # print('Resampled dataset shape %s' % Counter(y_res))
    return X_res


def check_args4plot(args):
    if 'box' in args.plot:
        args.plot_metric = ['aupr', 'auc', 'ndcg']
    return args


def plot_fig(args, ind, plot_set, params):
    args = check_args4plot(args)
    # First step: load data
    # prc, roc, box, hp, 3d, dist, pie, bar
    print(args.plot)
    for pl in args.plot:
        if pl in ['prc', 'roc']:
            metric = 'aupr' if pl == 'prc' else 'auc'
            au_dict = load_metric(args, ind, plot_set, metric, params)
            print('au_dict: ', au_dict)
            if plot_set == 'valid':
                true_y = np.load(args.data_dir + 'valid_y.npy') if not ind else np.load(args.data_dir + 'ind_y.npy')
            else:
                true_y = np.load(args.data_dir + 'test_y.npy') if not ind else np.load(args.data_dir + 'ind_y.npy')
            prob_dict = load_prob(args, ind, plot_set, params)
            if pl == 'prc':
                fig_path = args.fig_dir + 'prc.png' if not ind else args.fig_dir + 'ind_prc.png'
                plot_curve.plot_prc(true_y, prob_dict, au_dict, fig_path)
            else:
                fig_path = args.fig_dir + 'roc.png' if not ind else args.fig_dir + 'ind_roc.png'
                plot_curve.plot_roc(true_y, prob_dict, au_dict, fig_path)
        elif pl == 'box':
            aupr_list, auc_list, ndcg_list, labels = load_results(args, ind, plot_set, params)
            fig_path = args.fig_dir + 'box.png' if not ind else args.fig_dir + 'ind_box.png'
            plot_bin.box_fig(aupr_list, auc_list, ndcg_list, labels, fig_path)
        elif pl == 'polar':
            result_dict = load_eval(args, ind, plot_set, params)
            print(result_dict)
            fig_path = args.fig_dir + 'polar.png' if not ind else args.fig_dir + 'ind_polar.png'
            plot_bin.polar_fig(list(result_dict.keys()), list(result_dict.values()), params['metrics'], fig_path)
        elif pl == 'dr':
            prob_dict = load_prob(args, ind, plot_set, params)
            prob_arr = np.array(list(prob_dict.values()), dtype=np.float).transpose()
            if plot_set == 'valid':
                true_x = np.load(args.data_dir + 'valid_x.npy') if not ind else np.load(args.data_dir + 'ind_x.npy')
                true_y = np.load(args.data_dir + 'valid_y.npy') if not ind else np.load(args.data_dir + 'ind_y.npy')
            else:
                true_x = np.load(args.data_dir + 'test_x.npy') if not ind else np.load(args.data_dir + 'ind_x.npy')
                true_y = np.load(args.data_dir + 'test_y.npy') if not ind else np.load(args.data_dir + 'ind_y.npy')
            new_x, new_prob, new_y = data_rus(true_x, prob_arr, true_y)
            old_fig_path = args.fig_dir + 'old_dr_3d.png' if not ind else args.fig_dir + 'old_dr_3d_ind.png'
            plot_bin.plot_3d(new_x, new_y, old_fig_path, old=True)
            new_fig_path = args.fig_dir + 'new_dr_3d.png' if not ind else args.fig_dir + 'new_dr_3d_ind.png'
            plot_bin.plot_3d(new_prob, new_y, new_fig_path, old=False)
        elif pl == 'dist':
            args.rank = 'none'
            if plot_set == 'valid':
                true_y = np.load(args.data_dir + 'valid_y.npy') if not ind else np.load(args.data_dir + 'ind_y.npy')
            else:
                true_y = np.load(args.data_dir + 'test_y.npy') if not ind else np.load(args.data_dir + 'ind_y.npy')
            prob_dict = load_prob(args, ind, plot_set, params)
            new_prob = prob_rus(prob_dict, true_y)
            fig_path = args.fig_dir + 'dist.png' if not ind else args.fig_dir + 'dist_ind.png'
            plot_bin.dist_fig(new_prob, list(prob_dict.keys()), fig_path)
            args.rank = 'ltr'
        elif pl == 'hp':
            args.rank = 'none'
            prob_dict = load_prob(args, ind, plot_set, params)
            fig_path = args.fig_dir + 'hp.png' if not ind else args.fig_dir + 'hp_ind.png'
            plot_bin.hp_fig(pd.DataFrame(prob_dict), fig_path)
            args.rank = 'ltr'
        elif pl == 'pie':
            if args.rank == 'ltr':
                args.rank = 'none'
                prob_dict = load_prob(args, ind, plot_set, params)
                fig_path = args.fig_dir + 'pie.png' if not ind else args.fig_dir + 'pie_ind.png'
                plot_int.pie_fig(args.int_path + 'ltr_model.pkl', list(prob_dict.keys()), fig_path)
                args.rank = 'ltr'
            else:
                print('If you want to plot pie fig, the rank method ga or de should be selected!\n')
        elif pl == 'bar':
            if args.rank == 'ltr':
                args.rank = 'none'
                prob_dict = load_prob(args, ind, plot_set, params)
                fig_path = args.fig_dir + 'bar.png' if not ind else args.fig_dir + 'bar_ind.png'
                plot_int.bar_fig(args.int_path + 'ltr_model.pkl', list(prob_dict.keys()), fig_path)
                args.rank = 'ltr'
            else:
                print('If you want to plot bar fig, the rank method ltr should be selected!\n')
