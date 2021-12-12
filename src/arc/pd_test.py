import re

import numpy as np
import pandas as pd


def trans_text(str_data_list):
    res = []

    for str_data in str_data_list:
        str_list = re.findall('\d+', str_data)
        num_list = list(map(int, str_list))
        num_arr = np.array(num_list, dtype=np.float32)
        res.append(num_arr)

    return res


def trans_pd(file_name):
    pd_data = pd.read_csv(file_name)
    id_left_list = pd_data['id_left'].values
    text_left_list = trans_text(pd_data['text_left'].values)
    length_left_list = list(map(int, pd_data['length_left'].values))

    id_right_list = pd_data['id_right'].values
    text_right_list = trans_text(pd_data['text_right'].values)
    length_right_list = list(map(int, pd_data['length_right'].values))

    label_list = list(map(float, pd_data['label'].values))

    data = {'id_left': pd.Series(id_left_list),
            'text_left': pd.Series(text_left_list),
            'length_left': pd.Series(length_left_list),
            'id_right': pd.Series(id_right_list),
            'text_right': pd.Series(text_right_list),
            'length_right': pd.Series(length_right_list),
            'label': pd.Series(label_list)}

    return pd.DataFrame(data)
