#!/bin/bash
echo "Demo of web!"

python sesica_clf.py -base_dir ../data/web_homo -data_type homo -bmk_vec ../data/web_homo/bmk_vec.txt -bmk_label ../data/web_homo/bmk_pos_label.txt ../data/web_homo/bmk_neg_label.txt -ind score -ind_vec ../data/web_homo/ind_vec.txt -clf svm rf ert knn mnb gbdt goss dart mlp -metric auc

python sesica_rank.py -base_dir ../data/web_homo  -rank ltr -clf svm rf ert knn mnb gbdt goss dart mlp -metrics auc

python sesica_plot.py -base_dir ../data/web_homo -clf svm rf ert knn mnb gbdt goss dart mlp -rank ltr -plot roc prc