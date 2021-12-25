import os
import random


def path_check(target_dir):
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
            print("Path '" + target_dir + "' has been created!")
        except OSError:
            pass
    # else:
    #     # 先删除再创建
    #     try:
    #         shutil.rmtree(target_dir)
    #         os.makedirs(target_dir)
    #         print("Path '" + target_dir + "' has been created!")
    #     except OSError:
    #         pass
    return target_dir


def path_ctrl(args):
    args.base_dir = os.path.dirname(os.getcwd()) + '/'
    print("*Note that all files and models will be generated in the 'result' directory!!!\n")
    args.res_dir = args.base_dir + 'result/'
    # create directory for processed input data
    args.data_dir = path_check(args.res_dir + 'bmk_data/')
    # create directory for each classifier
    args.ssc_path = {}
    if args.clf != 'none':
        args.clf_path = {}
        for clf in args.clf:
            args.clf_path[clf] = path_check(args.res_dir + clf + '/')
            args.ssc_path[clf] = args.res_dir + clf + '/'
    # create directory for each deep-learning arc
    if args.arc != 'none':
        args.arc_path = {}
        for arc in args.arc:
            args.arc_path[arc] = path_check(args.res_dir + arc + '/')
            args.ssc_path[arc] = args.res_dir + arc + '/'

    if args.integrate == 'none':
        args.no_int_path = path_check(args.res_dir + 'no_int/')
    else:
        args.int_path = path_check(args.res_dir + args.integrate + '_int/')

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
    count = 0
    tmp_len = len(top_n)
    top_n_10 = {}
    for ml in clf:
        top_n_10[ml] = top_n[++count] if tmp_len >= count else 1
    return top_n_10


def make_clf_pk():
    return {'svm': ['svm_c', 'svm_g'], 'rf': ['rf_t'], 'ert': ['ert_t'], 'mnb': ['mnb_a'], 'bnb': ['bnb_a'],
            'gnb': ['none'], 'gbdt': ['gbdt_n', 'gbdt_t'], 'dart': ['dart_n', 'dart_t'], 'goss': ['goss_n', 'goss_t'],
            'mlp': ['act', 'hls']}
