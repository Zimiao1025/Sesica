import re

import matchzoo as mz
import numpy as np
import pandas as pd

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


def trans_text(str_data_list):
    res = []

    for str_data in str_data_list:
        str_list = re.findall('\d+', str_data)
        num_list = list(map(int, str_list))
        num_arr = np.array(num_list, dtype=np.float32)[:19]
        res.append(num_arr)

    return res


def trans_ngram(str_data_list):
    res = []

    for str_data in str_data_list:
        str_list = re.findall('\d+', str_data)
        num_list = list(map(int, str_list))
        num_arr = np.array(num_list, dtype=np.float32)[:18].reshape(3, 6)
        res.append(num_arr)

    return res


def trans_pd(file_name, arc):
    pd_data = pd.read_csv(file_name)
    id_left_list = pd_data['id_left'].values
    text_left_list = trans_text(pd_data['text_left'].values)
    ngram_left_list = trans_ngram(pd_data['text_left'].values)
    length_left_list = list(map(int, pd_data['length_left'].values))

    id_right_list = pd_data['id_right'].values
    text_right_list = trans_text(pd_data['text_right'].values)
    ngram_right_list = trans_ngram(pd_data['text_right'].values)
    length_right_list = list(map(int, pd_data['length_right'].values))

    label_list = list(map(float, pd_data['label'].values))

    if arc == 'dssm':
        left_vec = 'ngram_left'
        right_vec = 'ngram_right'
        data = {'id_left': pd.Series(id_left_list),
                'text_left': pd.Series(text_left_list),
                left_vec: pd.Series(text_left_list),
                'length_left': pd.Series(length_left_list),
                'id_right': pd.Series(id_right_list),
                'text_right': pd.Series(text_left_list),
                right_vec: pd.Series(text_right_list),
                'length_right': pd.Series(length_right_list),
                'label': pd.Series(label_list)}
    elif arc == 'cdssm':
        left_vec = 'ngram_left'
        right_vec = 'ngram_right'
        data = {'id_left': pd.Series(id_left_list),
                'text_left': pd.Series(text_left_list),
                left_vec: pd.Series(ngram_left_list),
                'length_left': pd.Series(length_left_list),
                'id_right': pd.Series(id_right_list),
                'text_right': pd.Series(text_left_list),
                right_vec: pd.Series(ngram_right_list),
                'length_right': pd.Series(length_right_list),
                'label': pd.Series(label_list)}
    else:
        data = {'id_left': pd.Series(id_left_list),
                'text_left': pd.Series(text_left_list),
                'length_left': pd.Series(length_left_list),
                'id_right': pd.Series(id_right_list),
                'text_right': pd.Series(text_left_list),
                'length_right': pd.Series(length_right_list),
                'label': pd.Series(label_list)}

    return pd.DataFrame(data)


def arc_preprocess(args, arc):
    # Prepare input data:
    train_data = trans_pd(args.data_dir + 'train_df.csv', arc)
    valid_data = trans_pd(args.data_dir + 'valid_df.csv', arc)

    train_processed = mz.pack(train_data)
    valid_processed = mz.pack(valid_data)
    # Generate pair-wise training data:
    train_set = mz.dataloader.Dataset(
        data_pack=train_processed,
        mode='pair',
        num_dup=1,
        num_neg=4,
        batch_size=32
    )
    valid_set = mz.dataloader.Dataset(
        data_pack=valid_processed,
        mode='point',
        batch_size=32
    )
    return train_set, valid_set


def arc_train(args):
    for arc in args.arc:
        train_set, valid_set = arc_preprocess(args, arc)
        if arc == 'anmm':
            anmm_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'arci':
            arci_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'arcii':
            arcii_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'bimpm':
            bimpm_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'dssm':
            dssm_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'cdssm':
            cdssm_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'conv_knrm':
            conv_knrm_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'diin':
            diin_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'drmm':
            drmm_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'drmmtks':
            drmmtks_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'duet':
            duet_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'esim':
            esim_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'hbmp':
            hbmp_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'knrm':
            knrm_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'match_lstm':
            match_lstm_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'match_pyramid':
            match_pyramid_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'match_srnn':
            match_srnn_train(train_set, valid_set, args.arc_path[arc])
        elif arc == 'mv_lstm':
            mv_lstm_train(train_set, valid_set, args.arc_path[arc])
