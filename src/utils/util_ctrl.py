import os
import random


def path_check(target_dir):
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
            print("Path '" + target_dir + "' has been created!")
        except OSError:
            pass
    else:
        # 先删除再创建
        try:
            import shutil
            shutil.rmtree(target_dir)
            os.makedirs(target_dir)
            print("Path '" + target_dir + "' has been created!")
        except OSError:
            pass
    return target_dir


def clf_path_ctrl(args):
    # args.base_dir = os.path.dirname(os.getcwd()) + '/'
    args.base_dir = os.path.abspath(args.base_dir) + '/'
    args.res_dir = args.base_dir + 'result/'
    # create directory for processed input data
    args.data_dir = path_check(args.res_dir + 'clf_data/')
    # create directory for each classifier
    if args.clf != 'none':
        # create directory for scale model
        args.scale_path = path_check(args.res_dir + 'scale/')
        args.clf_path = {}
        for clf in args.clf:
            args.clf_path[clf] = path_check(args.res_dir + clf + '/')
    return args


def arc_path_ctrl(args):
    # args.base_dir = os.path.dirname(os.getcwd()) + '/'
    args.base_dir = os.path.abspath(args.base_dir) + '/'
    args.res_dir = args.base_dir + 'result/'
    # create directory for processed input data
    args.data_dir = path_check(args.res_dir + 'arc_data/')
    # create directory for each deep-learning arc
    if args.arc != 'none':
        args.arc_path = {}
        for arc in args.arc:
            args.arc_path[arc] = path_check(args.res_dir + arc + '/')
    return args


def rank_path_ctrl(args):
    # args.base_dir = os.path.dirname(os.getcwd()) + '/'
    args.base_dir = os.path.abspath(args.base_dir) + '/'
    args.res_dir = args.base_dir + 'result/'
    # create directory for ranking
    if args.rank == 'none':
        if len(args.clf) >= 2:
            args.no_int_path = path_check(args.res_dir + 'no_int/')
    else:
        args.int_path = path_check(args.res_dir + args.rank + '_int/')
    return args


def plot_path_ctrl(args):
    """ bug1: 这里的path_check应该检查是否存在路径，如果不存在，直接报错，同rank
        bug2: args.data_dir
     """
    # args.base_dir = os.path.dirname(os.getcwd()) + '/'
    args.base_dir = os.path.abspath(args.base_dir) + '/'
    args.res_dir = args.base_dir + 'result/'
    args.data_dir = path_check(args.res_dir + 'clf_data/')
    args.res_dir = args.base_dir + 'result/'
    args.ssc_path = {}
    if args.clf != 'none':
        # create directory for scale model
        for clf in args.clf:
            args.ssc_path[clf] = path_check(args.res_dir + clf + '/')
    if args.arc != 'none':
        for arc in args.arc:
            args.ssc_path[arc] = path_check(args.res_dir + arc + '/')
    # create directory for plotting
    if args.plot != 'none':
        args.fig_dir = path_check(args.res_dir + 'plot/')

    return args


def sp_ctrl(associations):
    sp_associations = {}
    for key in associations.keys():
        if len(associations[key][0]) <= len(associations[key][1]):
            tmp_num = len(associations[key][0])
            random.seed(1025)
            sp_associations[key] = (associations[key][0], random.sample(associations[key][1], tmp_num))
        else:
            tmp_num = len(associations[key][1])
            random.seed(1025)
            sp_associations[key] = (random.sample(associations[key][0], tmp_num), associations[key][1])
    return sp_associations


def top_n_ctrl(clf, top_n):
    # top_n_order = [clf, arc]  ----  top --> list
    count = 0
    tmp_len = len(top_n)
    top_n_10 = {}
    if clf != 'none':
        for ml in clf:
            top_n_10[ml] = top_n[count] if count <= tmp_len - 1 else 1
            count += 1
    return top_n_10


def scale_ctrl(clf, scale):
    # scale = [mms, ss, ...,]  ----  scale --> list
    count = 0
    tmp_len = len(scale)
    scale_dict = {}
    if clf != 'none':
        for ml in clf:
            scale_dict[ml] = scale[count] if count <= tmp_len - 1 else 'none'
            count += 1
    # EXTRA
    if 'svm' in clf:
        scale_dict['svm'] = 'ss'
    if 'mlp' in clf:
        scale_dict['mlp'] = 'mms'
    if 'knn' in clf:
        scale_dict['knn'] = 'mms'
    return scale_dict


def make_clf_pk():
    # 不一定只是clf, 后续扩展
    return {'svm': ['svm_c', 'svm_g'], 'rf': ['rf_t'], 'ert': ['ert_t'], 'mnb': ['mnb_a'], 'knn': ['knn_n'],
            'gbdt': ['gbdt_n', 'gbdt_t'], 'dart': ['dart_n', 'dart_t'], 'goss': ['goss_n', 'goss_t'],
            'mlp': ['act', 'hls']}


def params_clf(args):
    params = {'scale': scale_ctrl(args.clf, args.scale), 'top_n': top_n_ctrl(args.clf, args.top_n),
              'metrics': args.metrics, 'clf_param_keys': make_clf_pk()}
    return params


def params_arc(args):
    params = {'metrics': args.metrics}
    return params


def params_rank(args):
    params = {'scale': scale_ctrl(args.clf, args.scale), 'top_n': top_n_ctrl(args.clf, args.top_n),
              'metrics': args.metrics, 'clf_param_keys': make_clf_pk()}
    return params


def params_plot(args):
    params = {'top_n': top_n_ctrl(args.clf, args.top_n)}
    return params
