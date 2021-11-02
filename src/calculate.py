import os
from datetime import datetime

from utils.dataset import read_csv
# from rank.ltr import train
# from rank.preprocess import load_data4rank
from score.process import score_process


def main(args):

    # 生成伪数据集并保存为csv
    csv_path = args.base_path + "/data/train/train_data.csv"
    # qk_dataset = gen_dataset(Sample_Num, Edge_Up_Bound, Fea_Len, Fea_Wid, csv_path)
    # exit()

    # 加载csv数据，并进行相似度计算
    csv_train_data = read_csv(csv_path)
    feats_path = args.base_path + "/data/train/train_feats.txt"
    group_path = args.base_path + "/data/train/train_group.txt"
    score_process(csv_train_data, args.score_methods, feats_path, group_path)

    # 加载 feats.txt 和 group.txt，输入 LTR
    # model_path = args.base_path + "/result/ltr_model/ltr.mod"
    # x_train, y_train, q_train = load_data4rank(feats_path, group_path)
    # train(x_train, y_train, q_train, model_path)


if __name__ == '__main__':
    Sample_Num = 5
    Edge_Up_Bound = 5
    Fea_Len = 2
    Fea_Wid = 2

    import argparse

    parse = argparse.ArgumentParser(prog='*', description="#")

    # parameters for whole framework
    parse.add_argument('-score_mode', type=str, choices=['U', 'R', 'I'],
                       help="The category of input sequences.")

    argv = parse.parse_args()

    argv.base_path = os.path.abspath(os.getcwd())
    # print(argv.base_path)
    # exit()

    train_start = datetime.now()
    print(" Analysis begin ".center(30, '*'))
    print("\n")

    argv.score_methods = ['ED', 'MD', 'CD', 'HD', 'JSC', 'CS', 'PCC']
    main(argv)

    print(" Analysis finish ".center(30, '*'))
    print("\n")
    train_end = datetime.now()
    consume_time = (train_end - train_start).seconds
    print("consume time : {}".format(consume_time))
