import sys
from itertools import count, takewhile


def svm_params_check(cost, gamma, gs_mode=0):  # 2: meticulous; 1: 'rough'; 0: 'none'.
    if gs_mode == 0:
        if len(cost) == 1:
            c_range = range(cost[0], cost[0] + 1, 1)
        elif len(cost) == 2:
            c_range = range(cost[0], cost[1] + 1, 1)
        elif len(cost) == 3:
            c_range = range(cost[0], cost[1] + 1, cost[2])
        else:
            error_info = 'The number of input value of parameter "cost" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    elif gs_mode == 1:
        c_range = range(-5, 11, 3)
    else:
        c_range = range(-5, 11, 1)

    if gs_mode == 0:
        if len(gamma) == 1:
            g_range = range(gamma[0], gamma[0] + 1, 1)
        elif len(gamma) == 2:
            g_range = range(gamma[0], gamma[1] + 1, 1)
        elif len(gamma) == 3:
            g_range = range(gamma[0], gamma[1] + 1, gamma[2])
        else:
            error_info = 'The number of input value of parameter "gamma" should be no more than 3!'
            sys.stderr.write(error_info)
            return False

    elif gs_mode == 1:
        g_range = range(-10, 6, 3)
    else:
        g_range = range(-10, 6, 1)

    return c_range, g_range


def rt_params_check(tree, gs_mode=0):
    if gs_mode == 0:
        if len(tree) == 1:
            t_range = range(tree[0], tree[0] + 100, 100)
        elif len(tree) == 2:
            t_range = range(tree[0], tree[1] + 100, 100)
        elif len(tree) == 3:
            t_range = range(tree[0], tree[1] + 100, tree[2])
        else:
            error_info = 'The number of input value of parameter "n_estimators" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    elif gs_mode == 1:
        t_range = range(100, 600, 100)
    else:
        t_range = range(100, 600, 200)

    return t_range


def f_range(start, stop, step):
    return takewhile(lambda x: x < stop, count(start, step))


def nb_params_check(alpha, gs_mode=0):
    if gs_mode == 0:
        if len(alpha) == 1:
            alpha_range = f_range(alpha[0], alpha[0] + 0.1, 0.1)
        elif len(alpha) == 2:
            alpha_range = f_range(alpha[0], alpha[1] + 0.1, 0.1)
        elif len(alpha) == 3:
            alpha_range = f_range(alpha[0], alpha[1] + 0.1, alpha[2])
        else:
            error_info = 'The number of input value of parameter "rf_tree" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    elif gs_mode == 1:
        alpha_range = f_range(0.0, 1.1, 0.3)
    else:
        alpha_range = f_range(0.0, 1.1, 0.1)

    return alpha_range


def lgb_params_check(tree, num_leaves, gs_mode=0):
    if gs_mode == 0:
        if len(tree) == 1:
            t_range = range(tree[0], tree[0] + 100, 100)
        elif len(tree) == 2:
            t_range = range(tree[0], tree[1] + 100, 100)
        elif len(tree) == 3:
            t_range = range(tree[0], tree[1] + 100, tree[2])
        else:
            error_info = 'The number of input value of parameter "rf_tree" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    elif gs_mode == 1:
        t_range = range(100, 600, 100)
    else:
        t_range = range(100, 600, 200)

    if gs_mode == 0:
        if len(num_leaves) == 1:
            n_range = range(num_leaves[0], num_leaves[0] + 100, 100)
        elif len(num_leaves) == 2:
            n_range = range(num_leaves[0], num_leaves[1] + 100, 100)
        elif len(num_leaves) == 3:
            n_range = range(num_leaves[0], num_leaves[1] + 100, num_leaves[2])
        else:
            error_info = 'The number of input value of parameter "rf_tree" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    elif gs_mode == 1:
        n_range = [31, 63, 127, 255, 511]
    else:
        n_range = [31, 63, 127]

    return t_range, n_range


def clf_params_control(clf, args, params):
    if clf == 'svm':
        params['cost'], params['gamma'] = svm_params_check(args.cost, args.gamma, args.gs_mode)
    elif clf in ['rf', 'et']:
        params['n_estimators'] = rt_params_check(args.n_estimators, args.gs_mode)
    elif clf in ['gnb', 'mnb', 'bnb']:
        params['nb_alpha'] = nb_params_check(args.nb_alpha, args.gs_mode)
    elif clf in ['gbdt', 'dart', 'goss']:
        params['lgb_tree'], params['num_leaves'] = lgb_params_check(args.n_estimators, args.num_leaves, args.gs_mode)
    else:
        # 不推荐进行遍历
        params['act'] = args.act
        params['hls'] = args.hls
    return params
