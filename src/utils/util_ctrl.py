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


def path_ctrl(args):
    args.base_dir = os.path.dirname(os.getcwd()) + '/'
    print("*Note that all files and models will be generated in the 'result' directory!!!\n")
    args.res_dir = args.base_dir + 'result/'
    # create directory for processed input data
    args.data_dir = args.res_dir + 'bmk_data/'
    path_check(args.data_dir)
    # create directory for each classifier
    if args.clf != 'none':
        args.clf_path = {}
        for clf in args.clf:
            args.clf_path[clf] = args.res_dir + clf + '/'
            path_check(args.clf_path[clf])

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


def file_ctrl(prefix_list, file_path):
    file_list = os.listdir(file_path)
    for prefix in prefix_list:
        for file_name in file_list:
            if prefix not in file_name:
                try:
                    os.remove(file_path+file_name)
                except OSError:
                    pass
                file_list.remove(file_name)
