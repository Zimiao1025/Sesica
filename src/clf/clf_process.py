import os

import joblib
import numpy as np
import pandas as pd

from clf.knn_impl import knn_train
from clf.lgb_impl import lgb_train
from clf.mlp_impl import mlp_train
from clf.nb_impl import nb_train
from clf.rt_impl import rt_train
from clf.svm_impl import svm_train
from utils import util_params, util_eval, util_data


def clf_train(args, params):
    print(' |Training| '.center(30, '*'))
    # training dataset
    train_x = np.load(args.data_dir + 'train_x.npy')
    train_y = np.load(args.data_dir + 'train_y.npy')
    # validation dataset
    valid_x = np.load(args.data_dir + 'valid_x.npy')
    valid_y = np.load(args.data_dir + 'valid_y.npy')
    valid_g = np.load(args.data_dir + 'valid_g.npy')

    for clf in args.clf:
        params = util_params.clf_params_control(clf, args, params)
        if params['scale'][clf] != 'none':
            train_x = util_data.pre_fit(train_x, params['scale'][clf], args.scale_path)
            valid_x = util_data.pre_trans(valid_x, params['scale'][clf], args.scale_path)

        if clf == 'svm':
            svm_train(train_x, train_y, valid_x, valid_y, valid_g, args.clf_path[clf], params)
        elif clf == 'knn':
            knn_train(train_x, train_y, valid_x, valid_y, valid_g, args.clf_path[clf], params)
        elif clf in ['rf', 'ert']:
            rt_train(clf, train_x, train_y, valid_x, valid_y, valid_g, args.clf_path[clf], params)
        elif clf == 'mnb':
            nb_train(train_x, train_y, valid_x, valid_y, valid_g, args.clf_path[clf], params)
        elif clf in ['gbdt', 'dart', 'goss']:
            lgb_train(clf, train_x, train_y, valid_x, valid_y, valid_g, args.clf_path[clf], params)
        elif clf == 'mlp':
            mlp_train(train_x, train_y, valid_x, valid_y, valid_g, args.clf_path[clf], params)


def clf_eval(args, params):
    print(' |Testing| '.center(30, '*'))
    test_x = np.load(args.data_dir + 'test_x.npy')
    test_y = np.load(args.data_dir + 'test_y.npy')
    test_g = np.load(args.data_dir + 'test_g.npy')
    prob_dict = {}
    for clf in args.clf:
        if params['scale'][clf] != 'none':
            test_x = util_data.pre_trans(test_x, params['scale'][clf], args.scale_path)
        model_path = args.clf_path[clf]
        for file_name in os.listdir(model_path):
            if file_name.endswith('pkl'):
                top_n = file_name.split('[')[1].split(']')[0]
                model = joblib.load(model_path + file_name)
                test_prob = model.predict_proba(test_x)[:, 1]
                prob_dict[top_n] = test_prob
                metric_df = util_eval.evaluation(params['metrics'], test_y, test_prob, test_g)
                metric_df.to_csv(model_path + top_n + '_test_results.csv')
                metric_list = metric_df.mean().tolist()
                print('Testing result of %s model: %s = %.4f\n' % (top_n, params['metrics'][0], metric_list[0]))

        df = pd.DataFrame(prob_dict)
        df.to_csv(model_path + 'test_prob.csv')


def clf_test(args, params):
    test_x = np.load(args.data_dir + 'ind_x.npy')
    test_y = np.load(args.data_dir + 'ind_y.npy')
    test_g = np.load(args.data_dir + 'ind_g.npy')
    prob_dict = {}
    for clf in args.clf:
        if params['scale'][clf] != 'none':
            test_x = util_data.pre_trans(test_x, params['scale'][clf], args.scale_path)
        model_path = args.clf_path[clf]
        for file_name in os.listdir(model_path):
            if file_name.endswith('pkl'):
                top_n = file_name.split('[')[1].split(']')[0]
                model = joblib.load(model_path + file_name)
                test_prob = model.predict_proba(test_x)[:, 1]
                prob_dict[top_n] = test_prob
                metric_df = util_eval.evaluation(params['metrics'], test_y, test_prob, test_g)
                metric_df.to_csv(model_path + top_n + '_ind_results.csv')
                metric_list = metric_df.mean().tolist()
                print('Independent test result of %s model: %s = %.4f\n' % (top_n, params['metrics'][0], metric_list[0]))

        df = pd.DataFrame(prob_dict)
        df.to_csv(model_path + 'ind_prob.csv')


def clf_score_hetero(args, params, tag='A'):
    test_x = np.load(args.data_dir + 'score_x.npy')
    prob_dict = {}
    score_prob = []
    for clf in args.clf:
        if params['scale'][clf] != 'none':
            test_x = util_data.pre_trans(test_x, params['scale'][clf], args.scale_path)
        model_path = args.clf_path[clf]
        for file_name in os.listdir(model_path):
            if file_name.endswith('pkl'):
                top_n = file_name.split('[')[1].split(']')[0]
                model = joblib.load(model_path + file_name)
                test_prob = model.predict_proba(test_x)[:, 1]
                prob_dict[top_n] = test_prob
                score_prob.append(test_prob)

        df = pd.DataFrame(prob_dict)
        df.to_csv(model_path + 'ind_score_prob.csv')
    ind_score = np.mean(np.array(score_prob), axis=0)
    print(ind_score)
    if tag == 'A':
        ind_data = pd.read_csv(args.data_dir + 'ind_vec_a.csv')
        bmk_data = pd.read_csv(args.data_dir + 'bmk_vec_b.csv')
        score_pair = np.load(args.data_dir + 'score_pair_A.npy')
    else:
        ind_data = pd.read_csv(args.data_dir + 'ind_vec_b.csv')
        bmk_data = pd.read_csv(args.data_dir + 'bmk_vec_a.csv')
        score_pair = np.load(args.data_dir + 'score_pair_B.npy')
    score_dict_list = []
    for idx, pair in enumerate(score_pair):
        query_idx, doc_idx = pair
        score_dict = {'query_index': query_idx, 'query_name': ind_data.loc[query_idx, 'vec_name'], 'doc_index': doc_idx,
                      'doc_name': bmk_data.loc[doc_idx, 'vec_name'], 'doc_vec': bmk_data.loc[doc_idx, 'vec_value'],
                      'score': ind_score[idx]}
        score_dict_list.append(score_dict)
    score_df = pd.DataFrame(score_dict_list)
    score_df.to_csv(args.score_dir + 'clf_score_results.csv')


def clf_score_homo(args, params):
    test_x = np.load(args.data_dir + 'score_x.npy')
    prob_dict = {}
    score_prob = []
    for clf in args.clf:
        if params['scale'][clf] != 'none':
            test_x = util_data.pre_trans(test_x, params['scale'][clf], args.scale_path)
        model_path = args.clf_path[clf]
        for file_name in os.listdir(model_path):
            if file_name.endswith('pkl'):
                top_n = file_name.split('[')[1].split(']')[0]
                model = joblib.load(model_path + file_name)
                test_prob = model.predict_proba(test_x)[:, 1]
                prob_dict[top_n] = test_prob
                score_prob.append(test_prob)

        df = pd.DataFrame(prob_dict)
        df.to_csv(model_path + 'ind_score_prob.csv')
    ind_score = np.mean(np.array(score_prob), axis=0)
    print(ind_score)
    ind_data = pd.read_csv(args.data_dir + 'ind_vec.csv')
    bmk_data = pd.read_csv(args.data_dir + 'bmk_vec.csv')
    score_pair = np.load(args.data_dir + 'score_pair.npy')

    score_dict_list = []
    for idx, pair in enumerate(score_pair):
        query_idx, doc_idx = pair
        score_dict = {'query_index': query_idx, 'query_name': ind_data.loc[query_idx, 'vec_name'], 'doc_index': doc_idx,
                      'doc_name': bmk_data.loc[doc_idx, 'vec_name'], 'doc_vec': bmk_data.loc[doc_idx, 'vec_value'],
                      'score': ind_score[idx]}
        score_dict_list.append(score_dict)
    score_df = pd.DataFrame(score_dict_list)
    score_df.to_csv(args.score_dir + 'clf_score_results.csv')
