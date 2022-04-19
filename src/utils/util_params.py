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
            error_info = 'The number of input value of parameter "svm_c" should be no more than 3!'
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
            error_info = 'The number of input value of parameter "svm_g" should be no more than 3!'
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
            error_info = 'The number of input value of parameter "rt_t/ert_t" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    elif gs_mode == 1:
        t_range = [64, 128, 256, 512, 1024]
    else:
        t_range = [32, 64, 128, 256, 512, 1024, 2048]

    return t_range


def knn_params_check(num, gs_mode=0):
    if gs_mode == 0:
        if len(num) == 1:
            n_range = range(num[0], num[0] + 10, 10)
        elif len(num) == 2:
            n_range = range(num[0], num[1] + 10, 10)
        elif len(num) == 3:
            n_range = range(num[0], num[1] + 10, num[2])
        else:
            error_info = 'The number of input value of parameter "knn_n" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    elif gs_mode == 1:
        n_range = [5, 10, 20, 50, 100]
    else:
        n_range = [1, 5, 10, 20, 50, 100, 200]

    return n_range


def f_range(start, stop, step):
    return takewhile(lambda x: x < stop, count(start, step))


def mnb_params_check(alpha, gs_mode=0):
    if gs_mode == 0:
        if len(alpha) == 1:
            alpha_range = f_range(alpha[0], alpha[0] + 0.1, 0.1)
        elif len(alpha) == 2:
            alpha_range = f_range(alpha[0], alpha[1] + 0.1, 0.1)
        elif len(alpha) == 3:
            alpha_range = f_range(alpha[0], alpha[1] + 0.1, alpha[2])
        else:
            error_info = 'The number of input value of parameter "mnb_a/bnb_a" should be no more than 3!'
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
            error_info = 'The number of input value of parameter "gbdt_t/dart_t/goss_t" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    elif gs_mode == 1:
        t_range = [64, 128, 256, 512, 1024]
    else:
        t_range = [32, 64, 128, 256, 512, 1024, 2048]

    if gs_mode == 0:
        if len(num_leaves) == 1:
            n_range = range(num_leaves[0], num_leaves[0] + 100, 100)
        elif len(num_leaves) == 2:
            n_range = range(num_leaves[0], num_leaves[1] + 100, 100)
        elif len(num_leaves) == 3:
            n_range = range(num_leaves[0], num_leaves[1] + 100, num_leaves[2])
        else:
            error_info = 'The number of input value of parameter "gbdt_n/dart_n/goss_n" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    elif gs_mode == 1:
        n_range = [31, 63, 127, 255, 511]
    else:
        n_range = [15, 31, 63, 127, 255, 511, 1025]

    return t_range, n_range


def clf_params_control(clf, args, params):
    if clf == 'svm':
        params['svm_c'], params['svm_g'] = svm_params_check(args.svm_c, args.svm_g, args.gs_mode)
    elif clf == 'rf':
        params['rf_t'] = rt_params_check(args.rf_t, args.gs_mode)
    elif clf == 'ert':
        params['ert_t'] = rt_params_check(args.ert_t, args.gs_mode)
    elif clf == 'knn':
        params['knn_n'] = knn_params_check(args.knn_n, args.gs_mode)
    elif clf == 'mnb':
        params['mnb_a'] = mnb_params_check(args.mnb_a, args.gs_mode)
    elif clf == 'gbdt':
        params['gbdt_t'], params['gbdt_n'] = lgb_params_check(args.gbdt_t, args.gbdt_n, args.gs_mode)
    elif clf == 'dart':
        params['dart_t'], params['dart_n'] = lgb_params_check(args.dart_t, args.dart_n, args.gs_mode)
    elif clf == 'goss':
        params['goss_t'], params['goss_n'] = lgb_params_check(args.goss_t, args.goss_n, args.gs_mode)
    else:
        # 不推荐进行遍历
        params['act'] = [args.act]
        params['hls'] = [tuple(args.hls)]
    return params


def arc_params_control(arc, args, params):
    params['num_neg'] = {}
    if arc == 'arci':
        params['num_neg']['arci'] = args.arci_neg
        params['arci_epoch'] = args.arci_epoch
        params['arci_dropout'] = args.arci_dropout
        params['arci_lr'] = args.arci_lr
        params['arci_layers'] = args.arci_layers
        params['arci_units'] = args.arci_units
        params['arci_emb_out'] = args.arci_emb
        if args.category == 'Protein':
            params['arci_emb_in'] = 21 ** args.word_size
        else:
            params['arci_emb_in'] = 5 ** args.word_size
    elif arc == 'arcii':
        params['num_neg']['arcii'] = args.arcii_neg
        params['arcii_epoch'] = args.arcii_epoch
        params['arcii_dropout'] = args.arcii_dropout
        params['arcii_lr'] = args.arcii_lr
        params['arcii_emb_out'] = args.arcii_emb
        if args.category == 'Protein':
            params['arcii_emb_in'] = 21 ** args.word_size
        else:
            params['arcii_emb_in'] = 5 ** args.word_size
    elif arc == 'dssm':
        params['num_neg']['dssm'] = args.dssm_neg
        params['ngram'] = args.dssm_ngram
        params['dssm_epoch'] = args.dssm_epoch
        params['dssm_dropout'] = args.dssm_dropout
        params['dssm_lr'] = args.dssm_lr
        params['dssm_layers'] = args.dssm_layers
        params['dssm_units'] = args.dssm_units
        # params['dssm_emb_out'] = args.dssm_emb
        params['vocab_size'] = args.fixed_len
    elif arc == 'cdssm':
        params['num_neg']['cdssm'] = args.cdssm_neg
        params['ngram'] = args.cdssm_ngram
        params['cdssm_epoch'] = args.cdssm_epoch
        params['cdssm_dropout'] = args.cdssm_dropout
        params['cdssm_lr'] = args.cdssm_lr
        params['cdssm_layers'] = args.cdssm_layers
        params['cdssm_units'] = args.cdssm_units
        # params['cdssm_emb_out'] = args.cdssm_emb
        params['vocab_size'] = args.fixed_len
    elif arc == 'drmm':
        params['num_neg']['drmm'] = args.drmm_neg
        params['drmm_epoch'] = args.drmm_epoch
        params['drmm_lr'] = args.drmm_lr
        params['drmm_layers'] = args.drmm_layers
        params['drmm_units'] = args.drmm_units
        params['drmm_emb_out'] = args.drmm_emb
        if args.category == 'Protein':
            params['drmm_emb_in'] = 21 ** args.word_size
        else:
            params['drmm_emb_in'] = 5 ** args.word_size
    return params


def ltr_params_check(max_depth, n_estimators, num_leaves, gs_mode=0):
    if gs_mode == 0:
        if len(max_depth) == 1:
            m_range = range(max_depth[0], max_depth[0] + 1, 1)
        elif len(max_depth) == 2:
            m_range = range(max_depth[0], max_depth[1] + 1, 1)
        elif len(max_depth) == 3:
            m_range = range(max_depth[0], max_depth[1] + 1, max_depth[2])
        else:
            error_info = 'The number of input value of parameter "ltr_t" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    elif gs_mode == 1:
        m_range = range(0, 7, 2)
    else:
        m_range = range(0, 10, 1)
    if gs_mode == 0:
        if len(n_estimators) == 1:
            t_range = range(n_estimators[0], n_estimators[0] + 100, 100)
        elif len(n_estimators) == 2:
            t_range = range(n_estimators[0], n_estimators[1] + 100, 100)
        elif len(n_estimators) == 3:
            t_range = range(n_estimators[0], n_estimators[1] + 100, n_estimators[2])
        else:
            error_info = 'The number of input value of parameter "ltr_t" should be no more than 3!'
            sys.stderr.write(error_info)
            return False
    elif gs_mode == 1:
        t_range = [64, 128, 256, 512, 1024]
    else:
        t_range = [32, 64, 128, 256, 512, 1024, 2048]

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
        n_range = [15, 31, 63, 127, 255, 511, 1025]

    return m_range, t_range, n_range


def rank_params_control(args, params):
    params['ltr_m'], params['ltr_t'], params['ltr_n'] = ltr_params_check(args.ltr_m, args.ltr_t, args.ltr_n,
                                                                             args.gs_mode)
    return params
