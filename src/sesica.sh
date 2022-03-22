#!/bin/bash
echo "Sesica !"
python sesica_clf.py -bmk_vec ../data/example/cir_fea.txt ../data/example/dis_fea.txt -bmk_label ../data/example/pos_pairs.txt ../data/example/neg_pairs.txt -clf svm
python sesica_arc.py -bmk_fasta ../data/mRNALoc/train.fasta -bmk_label ../data/mRNALoc/train_pos.txt ../data/mRNALoc/train_neg.txt -category DNA -arc arci -metrics auc