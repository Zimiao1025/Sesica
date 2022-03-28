#!/bin/bash
echo "Demo of web!"
python sesica_clf.py -base_dir ../data/web_demo -bmk_vec ../data/web_demo/bmk_vec_a.txt ../data/web_demo/bmk_vec_b.txt -bmk_label ../data/web_demo/bmk_pos_label.txt ../data/web_demo/bmk_neg_label.txt -clf svm rf ert knn mnb gbdt goss dart mlp -metrics auc aupr ndcg

python sesica_rank.py -base_dir ../data/web_demo  -clf svm rf ert knn mnb gbdt goss dart mlp -metrics auc aupr ndcg

python sesica_plot.py -base_dir ../data/web_demo -clf svm rf ert knn mnb gbdt goss dart mlp -rank ltr -plot roc prc