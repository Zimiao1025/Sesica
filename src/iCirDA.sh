#!/bin/bash
echo "identification of circRNA-disease associations!"
python sesica_clf.py -base_dir ../data/iCircDA -bmk_vec ../data/iCircDA/feature_3704.txt ../data/iCircDA/disSem_sim_90.txt -bmk_label ../data/iCircDA/benchmark_pos.txt ../data/iCircDA/benchmark_neg.txt -clf svm rf ert knn mnb gbdt goss dart mlp -metrics auc aupr ndcg ndcg@10 roc@10

python sesica_rank.py -base_dir ../data/iCircDA  -clf svm rf ert knn mnb gbdt goss dart mlp -metrics auc aupr ndcg

python sesica_plot.py -base_dir ../data/iCircDA -clf svm rf ert knn mnb gbdt goss dart mlp -rank ltr -plot roc prc