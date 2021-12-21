import os

import joblib
import numpy as np
import pandas as pd

from clf.lgb_impl import lgb_train
from clf.mlp_impl import mlp_train
from clf.nb_impl import nb_train
from clf.rt_impl import rt_train
from clf.svm_impl import svm_train
from utils import util_params, util_ctrl, util_eval


def clf_train(args):
    # training dataset
    train_x = np.load(args.data_dir + 'train_x.npy')
    train_y = np.load(args.data_dir + 'train_y.npy')
    # validation dataset
    valid_x = np.load(args.data_dir + 'valid_x.npy')
    valid_y = np.load(args.data_dir + 'valid_y.npy')
    valid_g = np.load(args.data_dir + 'valid_g.npy')

    params = {'top_n': util_ctrl.top_n_ctrl(args.clf, args.top_n), 'metrics': args.metrics}

    for clf in args.clf:
        params = util_params.clf_params_control(clf, args, params)

        if clf == 'svm':
            svm_train(train_x, train_y, valid_x, valid_y, valid_g, args.clf_path[clf], params)
        elif clf in ['rf', 'et']:
            rt_train(clf, train_x, train_y, valid_x, valid_y, valid_g, args.clf_path[clf], params)
        elif clf in ['gnb', 'mnb', 'bnb']:
            nb_train(clf, train_x, train_y, valid_x, valid_y, valid_g, args.clf_path[clf], params)
        elif clf in ['gbdt', 'dart', 'goss']:
            lgb_train(clf, train_x, train_y, valid_x, valid_y, valid_g, args.clf_path[clf], params)
        else:
            mlp_train(train_x, train_y, valid_x, valid_y, valid_g, args.clf_path[clf], params)


def clf_test(args):
    test_x = np.load(args.data_dir + 'test_x.npy')
    test_y = np.load(args.data_dir + 'test_y.npy')
    test_g = np.load(args.data_dir + 'test_g.npy')
    prob_dict = {}
    for clf in args.clf:
        model_path = args.clf_path[clf]
        for file_name in os.listdir(model_path):
            if file_name.endswith('pkl'):
                top_n = file_name.split('[')[1].split(']')[0]
                model = joblib.load(model_path + file_name)
                test_prob = model.predict_proba(test_x)[:, 1]
                prob_dict[top_n] = test_prob
                metric_list = util_eval.evaluation(args.metrics, test_y, test_prob, test_g,
                                                   model_path + os.path.splitext(file_name)[0] + '_test_eval.csv')
                print(' Test '.center(36, '*'))
                print('Evaluation on test dataset: ', metric_list[0])
                print('\n')

        df = pd.DataFrame(prob_dict)
        df.to_csv(model_path + 'test_prob.csv')


def clf_predict(args):
    test_x = np.load(args.data_dir + 'ind_x.npy')
    test_y = np.load(args.data_dir + 'ind_y.npy')
    test_g = np.load(args.data_dir + 'ind_g.npy')
    prob_dict = {}
    for clf in args.clf:
        model_path = args.clf_path[clf]
        for file_name in os.listdir(model_path):
            if file_name.endswith('pkl'):
                top_n = file_name.split('[')[1].split(']')[0]
                model = joblib.load(model_path + file_name)
                test_prob = model.predict_proba(test_x)[:, 1]
                prob_dict[top_n] = test_prob
                metric_list = util_eval.evaluation(args.metrics, test_y, test_prob, test_g,
                                                   model_path + os.path.splitext(file_name)[0] + '_ind_eval.csv')
                print(' Ind Test '.center(36, '*'))
                print('Evaluation on independent test dataset: ', metric_list[0])
                print('\n')

        df = pd.DataFrame(prob_dict)
        df.to_csv(model_path + 'ind_prob.csv')
