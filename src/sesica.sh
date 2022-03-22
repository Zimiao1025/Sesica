#!/bin/bash
echo "Sesica !"
python sesica_clf.py -base_dir ../data/web_demo -bmk_vec ../data/web_demo/cir_fea.txt ../data/web_demo/dis_fea.txt -bmk_label ../data/web_demo/pos_pairs.txt ../data/web_demo/neg_pairs.txt -clf svm