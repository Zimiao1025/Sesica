#!/bin/bash
echo "GO!"
python sesica_clf.py -base_dir ../data/go -data_type hetero -bmk_vec_a ../data/go/cc_bmk_vec_a.txt -bmk_vec_b ../data/go/cc_bmk_vec_b.txt -bmk_label ../data/go/pos_label.txt ../data/go/neg_label.txt -clf svm rf ert knn mnb gbdt goss dart mlp -metric aupr -gs_mode 2

python sesica_rank.py -base_dir ../data/go -rank ltr -clf svm rf ert knn mnb gbdt goss dart mlp -metric auc -gs_mode 2

python sesica_plot.py -base_dir ../data/go -clf svm rf ert knn mnb gbdt goss dart mlp -rank ltr -plot roc prc box polar hp dr dist pie bar -plot_set test