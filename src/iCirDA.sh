#!/bin/bash
echo "identification of circRNA-disease associations!"
python sesica_clf.py -base_dir ../data/web_demo -data_type hetero -bmk_vec_a ../data/web_demo/bmk_vec_a.txt -bmk_vec_b ../data/web_demo/bmk_vec_b.txt -bmk_label ../data/web_demo/bmk_pos_label.txt ../data/web_demo/bmk_neg_label.txt -ind score -ind_vec_a ../data/web_demo/ind_vec_a.txt -clf svm rf ert knn mnb gbdt goss dart mlp -metric auc

python sesica_rank.py -base_dir ../data/iCircDA  -clf svm rf ert knn mnb gbdt goss dart mlp -metrics auc aupr ndcg

python sesica_plot.py -base_dir ../data/iCircDA -clf svm rf ert knn mnb gbdt goss dart mlp -rank ltr -plot roc prc