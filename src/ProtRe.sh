#!/bin/bash
echo "Protein Recognition!"
python sesica_clf.py -base_dir ../data/ProtRe -data_type homo -bmk_vec ../data/ProtRe/bmk_vec.txt -bmk_label ../data/ProtRe/pos_label.txt ../data/ProtRe/neg_label.txt -clf svm rf ert knn mnb gbdt goss dart mlp -metric roc@1 -gs_mode 2

python sesica_arc.py -base_dir ../data/ProtRe -category Protein -bmk_fasta ../data/ProtRe/bmk_fasta.fasta -bmk_label ../data/ProtRe/pos_label.txt ../data/ProtRe/neg_label.txt -arc dssm -word_size 3 -metric roc@1 -dssm_epoch 10 -fixed_len 300
python sesica_arc.py -base_dir ../data/ProtRe -category Protein -bmk_fasta ../data/ProtRe/bmk_fasta.fasta -bmk_label ../data/ProtRe/pos_label.txt ../data/ProtRe/neg_label.txt -arc cdssm -word_size 3 -metric roc@1 -cdssm_epoch 5 -fixed_len 300
python sesica_arc.py -base_dir ../data/ProtRe -category Protein -bmk_fasta ../data/ProtRe/bmk_fasta.fasta -bmk_label ../data/ProtRe/pos_label.txt ../data/ProtRe/neg_label.txt -arc arci -word_size 2 -metric roc@1 -arci_epoch 5 -fixed_len 300
python sesica_arc.py -base_dir ../data/ProtRe -category Protein -bmk_fasta ../data/ProtRe/bmk_fasta.fasta -bmk_label ../data/ProtRe/pos_label.txt ../data/ProtRe/neg_label.txt -arc arcii -word_size 2 -metric roc@1 -arcii_epoch 5 -fixed_len 300
python sesica_arc.py -base_dir ../data/ProtRe -category Protein -bmk_fasta ../data/ProtRe/bmk_fasta.fasta -bmk_label ../data/ProtRe/pos_label.txt ../data/ProtRe/neg_label.txt -arc drmm -word_size 3 -metric roc@1 -drmm_epoch 5 -fixed_len 300


python sesica_rank.py -base_dir ../data/ProtRe -rank ltr -clf svm rf ert knn mnb gbdt goss dart mlp -metric auc -gs_mode 2

python sesica_plot.py -base_dir ../data/ProtRe -clf svm rf ert knn mnb gbdt goss dart mlp -rank ltr -plot roc prc box polar hp dr dist pie bar -plot_set test