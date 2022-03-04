import os

import joblib
import numpy as np
import pandas as pd

from rank.ltr_impl import ltr_train
from utils import util_params, util_eval


def rank_out(args):
    opt_list = []
    opt_val = 0.0
    for clf in args.clf:
        metric_df = pd.read_csv(args.ssc_path[clf] + 'top_1_eval_results.csv', dtype=np.float)
        metric_list = metric_df.mean().tolist()
        if opt_val <= metric_list[0]:
            opt_list = metric_list
            opt_val = metric_list[0]

    metric_data = {'metric': list(args.metric.keys()), 'val': opt_list}
    metric_pd = pd.DataFrame(metric_data)
    metric_pd.to_csv(args.no_int_path + 'final_result.csv')


def rank_out_ind(args):
    opt_list = []
    opt_val = 0.0
    for clf in args.clf:
        metric_df = pd.read_csv(args.ssc_path[clf] + 'top_1_eval_results_ind.csv', dtype=np.float)
        metric_list = metric_df.mean().tolist()
        if opt_val <= metric_list[0]:
            opt_list = metric_list
            opt_val = metric_list[0]

    metric_data = {'metric': list(args.metric.keys()), 'val': opt_list}
    metric_pd = pd.DataFrame(metric_data)
    metric_pd.to_csv(args.no_int_path + 'final_result_ind.csv')


def int_train(args, params):
    train_x = []
    valid_x = []
    for clf in args.clf:
        model_path = args.ssc_path[clf]
        valid_prob = pd.read_csv(model_path + 'valid_prob.csv', dtype=np.float)
        train_x.append(valid_prob.values[:, 1:].flatten())
        test_prob = pd.read_csv(model_path + 'test_prob.csv', dtype=np.float)
        valid_x.append(test_prob.values[:, 1:].flatten())

    # training dataset
    train_x = np.array(train_x).transpose()
    np.save(args.int_path + 'int_train_x.npy', train_x)
    train_y = np.load(args.data_dir + 'valid_y.npy')
    train_g = np.load(args.data_dir + 'valid_g.npy')
    # validation dataset
    valid_x = np.array(valid_x).transpose()
    np.save(args.int_path + 'int_valid_x.npy', valid_x)
    valid_y = np.load(args.data_dir + 'test_y.npy')
    valid_g = np.load(args.data_dir + 'test_g.npy')

    print('Start integration......\n')
    int_method = args.integrate

    params = util_params.int_params_control(int_method, args, params)

    if int_method == 'ltr':
        ltr_train(train_x, train_y, train_g, valid_x, valid_y, valid_g, args.int_path, params)


def int_predict(args):
    test_x = np.load(args.data_dir + 'ind_x.npy')
    test_y = np.load(args.data_dir + 'ind_y.npy')
    test_g = np.load(args.data_dir + 'ind_g.npy')
    prob_dict = {}
    for clf in args.clf:
        model_path = args.ssc_path[clf]
        for file_name in os.listdir(model_path):
            if file_name.endswith('pkl'):
                top_n = file_name.split('[')[1].split(']')[0]
                model = joblib.load(model_path + file_name)
                test_prob = model.predict_proba(test_x)[:, 1]
                prob_dict[top_n] = test_prob
                metric_list = util_eval.evaluation(args.metrics, test_y, test_prob, test_g)
                print(' Ind Test '.center(36, '*'))
                print('Evaluation on independent test dataset: ', metric_list[0])
                print('\n')

        df = pd.DataFrame(prob_dict)
        df.to_csv(model_path + 'ind_prob.csv')


def int_or_rank(args, params):
    if args.integrate == 'none':
        rank_out(args)
    else:
        int_train(args, params)


def int_or_rank_ind(args):
    if args.integrate == 'none':
        rank_out_ind(args)
    else:
        int_predict(args)
