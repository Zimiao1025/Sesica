import os

import joblib
import numpy as np
import pandas as pd

from integrate.lr_impl import lr_train
from integrate.ltr_impl import ltr_train
from integrate.wm_impl import weight_mean
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


def int_train(args):
    train_x = []
    valid_x = []
    for clf in args.clf:
        model_path = args.ssc_path[clf]
        valid_prob = pd.read_csv(model_path + 'valid_prob.csv', dtype=np.float)
        train_x.append(valid_prob)
        test_prob = pd.read_csv(model_path + 'test_prob.csv', dtype=np.float)
        valid_x.append(test_prob)

    # training dataset
    train_x = np.array(train_x)
    train_y = np.load(args.data_dir + 'valid_y.npy')
    train_g = np.load(args.data_dir + 'valid_g.npy')
    # validation dataset
    valid_x = np.array(valid_x)
    valid_y = np.load(args.data_dir + 'test_y.npy')
    valid_g = np.load(args.data_dir + 'test_g.npy')

    print('Start integration......\n')
    int_method = args.integrate
    params = {'metrics': args.metrics}

    params = util_params.int_params_control(int_method, args, params)

    if int_method in ['de', 'ga']:
        weight_mean(int_method, train_x, train_y, train_g, args.int_path[int_method], params)
    elif int_method == 'lr':
        lr_train(train_x, train_y, valid_x, valid_y, valid_g, args.int_path[int_method], params)
    elif int_method == 'ltr':
        ltr_train(train_x, train_y, train_g, valid_x, valid_y, valid_g, args.int_path[int_method], params)
    # else:


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


def int_or_rank(args):
    if args.integrate == 'none':
        rank_out(args)
    else:
        int_train(args)


def int_or_rank_ind(args):
    if args.integrate == 'none':
        rank_out_ind(args)
    else:
        int_predict(args)
