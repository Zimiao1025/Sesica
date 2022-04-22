#!/bin/bash
echo "Protein Recognition!"
python sesica_clf.py -base_dir ../data/ProtRe -data_type homo -bmk_vec ../data/ProtRe/bmk_vec.txt -bmk_label ../data/ProtRe/pos_label.txt ../data/ProtRe/neg_label.txt -clf svm rf ert knn mnb gbdt goss dart mlp -metric roc@1 -gs_mode 2

python sesica_rank.py -base_dir ../data/ProtRe -rank ltr -clf svm rf ert knn mnb gbdt goss dart mlp -metric auc -gs_mode 2

python sesica_plot.py -base_dir ../data/ProtRe -clf svm rf ert knn mnb gbdt goss dart mlp -rank ltr -plot roc prc box polar hp dr dist pie bar -plot_set test