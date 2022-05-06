import re

import matchzoo as mz
import numpy as np
import pandas as pd

from arc._common import prob_metric_cal
from arc.anmm_impl import anmm_train
from arc.arci_impl import arci_train
from arc.arcii_impl import arcii_train
from arc.bimpm_impl import bimpm_train
from arc.cdssm_impl import cdssm_train
from arc.conv_knrm_impl import conv_knrm_train
from arc.diin_impl import diin_train
from arc.drmm_impl import drmm_train
from arc.drmmtks_impl import drmmtks_train
from arc.dssm_impl import dssm_train
from arc.duet_impl import duet_train
from arc.esim_impl import esim_train
from arc.hbmp_impl import hbmp_train
from arc.knrm_impl import knrm_train
from arc.match_lstm_impl import match_lstm_train
from arc.match_pyramid_impl import match_pyramid_train
from arc.match_srnn_impl import match_srnn_train
from arc.mv_lstm_impl import mv_lstm_train

from utils.util_params import arc_params_control


def trans_text(str_data_list):
    res = []

    for str_data in str_data_list:
        str_list = re.findall('\d+', str_data)
        num_list = list(map(int, str_list))
        num_arr = np.array(num_list, dtype=np.float32)
        res.append(num_arr)
    print('Shape of text: ', np.array(res).shape)
    return res


def trans_ngram(str_data_list, ngram=3):
    res = []

    for str_data in str_data_list:
        str_list = re.findall('\d+', str_data)
        num_list = list(map(int, str_list))
        num_arr = []
        for i in range(len(num_list)):
            if i < len(num_list) - ngram + 1:
                gram = num_list[i: i + ngram]
            else:
                gram = num_list[i: len(num_list)] + [0] * (ngram - (len(num_list) - i))
            num_arr.append(gram)
        res.append(np.array(num_arr, dtype=np.float))
    print('Shape of n-gram: ', np.array(res).shape)
    return res


def trans_hist(str_data_list_left, str_data_list_right, bin_size):

    res_left = trans_ngram(str_data_list_left, 5)
    res_right = trans_ngram(str_data_list_right, 5)
    res_len = len(res_right[0])
    for left_text, right_text in zip(res_left, res_right):
        for i in range(res_len):
            score_list = []
            for j in range(res_len):
                score = np.dot(left_text[i], right_text[j]) / (np.linalg.norm(left_text[i]) * (np.linalg.norm(right_text[j])))
                score_list.append(score)

    # print('Shape of n-gram: ', np.array(res).shape)
    # return res


def trans_pd(file_name, arc, params):
    pd_data = pd.read_csv(file_name)

    id_left_list = pd_data['id_left'].values
    text_left_list = trans_text(pd_data['text_left'].values)
    length_left_list = list(map(int, pd_data['length_left'].values))

    id_right_list = pd_data['id_right'].values
    text_right_list = trans_text(pd_data['text_right'].values)
    length_right_list = list(map(int, pd_data['length_right'].values))

    label_list = list(map(float, pd_data['label'].values))

    if arc == 'dssm':
        # ngram_left_list = trans_ngram(pd_data['text_left'].values, params['ngram'])
        # ngram_right_list = trans_ngram(pd_data['text_right'].values, params['ngram'])
        data = {'id_left': pd.Series(id_left_list),
                'text_left': pd.Series(text_left_list),
                'ngram_left': pd.Series(text_left_list),
                'length_left': pd.Series(length_left_list),
                'id_right': pd.Series(id_right_list),
                'text_right': pd.Series(text_right_list),
                'ngram_right': pd.Series(text_right_list),
                'length_right': pd.Series(length_right_list),
                'label': pd.Series(label_list)}
    elif arc in ['cdssm', 'duet']:
        ngram_left_list = trans_ngram(pd_data['text_left'].values, params['ngram'])
        ngram_right_list = trans_ngram(pd_data['text_right'].values, params['ngram'])
        data = {'id_left': pd.Series(id_left_list),
                'text_left': pd.Series(text_left_list),
                'ngram_left': pd.Series(ngram_left_list),
                'length_left': pd.Series(length_left_list),
                'id_right': pd.Series(id_right_list),
                'text_right': pd.Series(text_right_list),
                'ngram_right': pd.Series(ngram_right_list),
                'length_right': pd.Series(length_right_list),
                'label': pd.Series(label_list)}
    else:
        data = {'id_left': pd.Series(id_left_list),
                'text_left': pd.Series(text_left_list),
                'length_left': pd.Series(length_left_list),
                'id_right': pd.Series(id_right_list),
                'text_right': pd.Series(text_right_list),
                'length_right': pd.Series(length_right_list),
                'label': pd.Series(label_list)}

    return pd.DataFrame(data)


def arc_train_preprocess(args, arc, params):
    # Prepare input data:
    train_data = trans_pd(args.data_dir + 'train_df.csv', arc, params)
    valid_data = trans_pd(args.data_dir + 'valid_df.csv', arc, params)
    test_data = trans_pd(args.data_dir + 'test_df.csv', arc, params)

    train_processed = mz.pack(train_data)
    valid_processed = mz.pack(valid_data)
    test_processed = mz.pack(test_data)
    # Generate pair-wise training data:
    train_set = mz.dataloader.Dataset(
        data_pack=train_processed,
        mode='pair',
        num_dup=1,
        num_neg=params['num_neg'][arc],
        batch_size=32
    )
    valid_set = mz.dataloader.Dataset(
        data_pack=valid_processed,
        mode='point',
        batch_size=32
    )
    test_set = mz.dataloader.Dataset(
        data_pack=test_processed,
        mode='point',
        batch_size=32
    )
    if args.ind:
        ind_data = trans_pd(args.data_dir + 'ind_df.csv', arc, params)
        ind_processed = mz.pack(ind_data)
        ind_set = mz.dataloader.Dataset(
            data_pack=ind_processed,
            mode='point',
            batch_size=32
        )
    else:
        ind_set = None

    return train_set, valid_set, test_set, ind_set


def load_label_group(args):
    valid_y = np.load(args.data_dir + 'valid_y.npy')
    valid_g = np.load(args.data_dir + 'valid_g.npy')

    test_y = np.load(args.data_dir + 'test_y.npy')
    test_g = np.load(args.data_dir + 'test_g.npy')

    if args.ind:
        ind_y = np.load(args.data_dir + 'ind_y.npy')
        ind_g = np.load(args.data_dir + 'ind_g.npy')
    else:
        ind_y = None
        ind_g = None

    return valid_y, valid_g, test_y, test_g, ind_y, ind_g


def arc_ctrl(args, params):
    valid_y, valid_g, test_y, test_g, ind_y, ind_g = load_label_group(args)
    for arc in args.arc:
        params = arc_params_control(arc, args, params)

        if arc == 'anmm':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = anmm_train(train_set, valid_set, test_set,
                                                                        args.arc_path[arc], ind_set, params)
        elif arc == 'arci':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = arci_train(train_set, valid_set, test_set,
                                                                        args.arc_path[arc], ind_set, params)
        elif arc == 'arcii':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = arcii_train(train_set, valid_set, test_set,
                                                                         args.arc_path[arc], ind_set, params)
        elif arc == 'bimpm':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = bimpm_train(train_set, valid_set, test_set,
                                                                         args.arc_path[arc], ind_set, params)
        elif arc == 'dssm':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = dssm_train(train_set, valid_set, test_set,
                                                                        args.arc_path[arc], ind_set, params)
        elif arc == 'cdssm':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = cdssm_train(train_set, valid_set, test_set,
                                                                         args.arc_path[arc], ind_set, params)
        elif arc == 'conv_knrm':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = conv_knrm_train(train_set, valid_set, test_set,
                                                                             args.arc_path[arc], ind_set, params)
        elif arc == 'diin':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = diin_train(train_set, valid_set, test_set,
                                                                        args.arc_path[arc], ind_set, params)
        elif arc == 'drmm':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = drmm_train(train_set, valid_set, test_set,
                                                                        args.arc_path[arc], ind_set, params)
        elif arc == 'drmmtks':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = drmmtks_train(train_set, valid_set, test_set,
                                                                           args.arc_path[arc], ind_set, params)
        elif arc == 'duet':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = duet_train(train_set, valid_set, test_set,
                                                                        args.arc_path[arc], ind_set, params)
        elif arc == 'esim':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = esim_train(train_set, valid_set, test_set,
                                                                        args.arc_path[arc], ind_set, params)
        elif arc == 'hbmp':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = hbmp_train(train_set, valid_set, test_set,
                                                                        args.arc_path[arc], ind_set, params)
        elif arc == 'knrm':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = knrm_train(train_set, valid_set, test_set,
                                                                        args.arc_path[arc], ind_set, params)
        elif arc == 'match_lstm':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = match_lstm_train(train_set, valid_set, test_set,
                                                                              args.arc_path[arc], ind_set, params)
        elif arc == 'match_pyramid':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = match_pyramid_train(train_set, valid_set, test_set,
                                                                                 args.arc_path[arc], ind_set, params)
        elif arc == 'match_srnn':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = match_srnn_train(train_set, valid_set, test_set,
                                                                              args.arc_path[arc], ind_set, params)
        elif arc == 'mv_lstm':
            train_set, valid_set, test_set, ind_set = arc_train_preprocess(args, arc, params)
            trainer, valid_loader, test_loader, ind_loader = mv_lstm_train(train_set, valid_set, test_set,
                                                                           args.arc_path[arc], ind_set, params)
        else:
            print('Arc error')
            # trainer, valid_loader, test_loader, ind_loader = None, None, None, None
            return False

        # calculate the predict prob and evaluation result
        prob_metric_cal(trainer, args.arc_path[arc], valid_y, valid_loader, valid_g, test_y, test_loader, test_g,
                        ind_loader, ind_y, ind_g, params)
