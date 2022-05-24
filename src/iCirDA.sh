#!/bin/bash
echo "identification of circRNA-disease associations!"
python sesica_clf.py -base_dir ../data/iCircDA -data_type hetero -bmk_vec_a ../data/iCircDA/bmk_circRNA.txt -bmk_vec_b ../data/iCircDA/bmk_disease.txt -bmk_label ../data/iCircDA/benchmark_pos.txt ../data/iCircDA/benchmark_neg.txt -clf svm rf ert knn mnb gbdt goss mlp -metric auc -gs_mode 2

python sesica_rank.py -base_dir ../data/iCircDA -rank ltr -clf svm rf ert knn mnb gbdt goss mlp -metric auc -gs_mode 2

python sesica_rank.py -base_dir ../data/iCircDA -rank ltr -clf svm rf knn mnb goss -metric auc -gs_mode 2

python sesica_plot.py -base_dir ../data/iCircDA -data_type hetero -clf svm rf ert knn mnb gbdt goss mlp -rank ltr -plot roc prc box polar hp dr dist pie bar -plot_set test