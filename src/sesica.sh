#!/bin/bash
echo "Sesica !"
python sesica_arc.py -bmk_fasta ../data/mRNALoc/train.fasta -bmk_label ../data/mRNALoc/train_pos.txt ../data/mRNALoc/train_neg.txt -category DNA -arc arci -metrics auc