#!/bin/bash
echo "Demo of web!"

python sesica_clf.py -base_dir ../data/web_hetero -data_type hetero -bmk_vec_a ../data/web_hetero/bmk_vec_a.txt -bmk_vec_b ../data/web_hetero/bmk_vec_b.txt -bmk_label ../data/web_hetero/bmk_pos_label.txt ../data/web_hetero/bmk_neg_label.txt -ind score -ind_vec_a ../data/web_hetero/ind_vec_a.txt -clf svm rf ert knn mnb gbdt goss dart mlp -metric auc

python sesica_rank.py -base_dir ../data/web_hetero -rank ltr -clf svm rf ert knn mnb gbdt goss dart mlp -metric auc

python sesica_plot.py -base_dir ../data/web_hetero -clf svm rf ert knn mnb gbdt goss dart mlp -rank ltr -plot roc prc