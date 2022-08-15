# BioSeq-Diabolo
biological sequence similarity analysing using Diabolo.

**Software Requirements:**

* [Python3](https://docs.python-guide.org/starting/install3/linux/)
* [virtualenv](https://virtualenv.pypa.io/en/latest/installation/) or [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) (Optional If using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional If using GPU)

BioSeq-Diabolo has been tested on Windows, Ubuntu 16.04, and 18.04 operating systems.

## Installation

### virtualenv

```shell
virtualenv -p python3.7 venv

source ./venv/bin/activate

# You can use 'requirements' file:
pip install -r requirements.txt
# or directly install the corresponding package:
pip install matchzoo-py
pip install scikit-learn
pip install lightgbm
pip install seaborn
```

### Anaconda

```shell
conda create -n venv python=3.7

conda activate venv

# You can use 'requirements' file:
pip install -r requirements.txt
# or directly install the corresponding package:
pip install matchzoo-py
pip install scikit-learn
pip install lightgbm
pip install seaborn
```

## Command

### Demo of web server
```python
python sesica_clf.py -base_dir ../data/web_hetero -data_type hetero -bmk_vec_a ../data/web_hetero/bmk_vec_a.txt -bmk_vec_b ../data/web_hetero/bmk_vec_b.txt -bmk_label ../data/web_hetero/bmk_pos_label.txt ../data/web_hetero/bmk_neg_label.txt -ind score -ind_vec_a ../data/web_hetero/ind_vec_a.txt -clf svm rf ert knn mnb gbdt goss dart mlp -metric aupr

python sesica_rank.py -base_dir ../data/web_hetero -rank ltr -clf svm rf ert knn mnb gbdt goss dart mlp -rank ltr -metric aupr

python sesica_plot.py -base_dir ../data/web_hetero -data_type hetero -clf svm rf ert knn mnb gbdt goss dart mlp -plot pie net roc prc box dist dr hp
```

### Identification of circRNA-disease associations
```python
python sesica_clf.py -base_dir ../data/iCircDA -data_type hetero -bmk_vec_a ../data/iCircDA/bmk_circRNA.txt -bmk_vec_b ../data/iCircDA/bmk_disease.txt -bmk_label ../data/iCircDA/benchmark_pos.txt ../data/iCircDA/benchmark_neg.txt -clf svm rf ert knn mnb gbdt goss mlp -metric auc -gs_mode 2

python sesica_rank.py -base_dir ../data/iCircDA -rank ltr -clf svm rf knn mnb goss -metric auc -gs_mode 2

python sesica_plot.py -base_dir ../data/iCircDA -data_type hetero -clf svm rf knn mnb goss -rank ltr -plot roc polar hp dr pie -plot_set test
```

### Protein remote homology detection
```python
python sesica_clf.py -base_dir ../data/ProtRe -data_type homo -bmk_vec ../data/ProtRe/bmk_vec.txt -bmk_label ../data/ProtRe/pos_label.txt ../data/ProtRe/neg_label.txt -clf svm rf ert knn gbdt goss dart mlp -metric roc@1 -gs_mode 2

python sesica_rank.py -base_dir ../data/ProtRe -rank ltr -clf svm rf ert knn mlp -metric roc@1 -gs_mode 2

python sesica_plot.py -base_dir ../data/ProtRe -data_type homo -clf svm rf ert knn mlp -rank ltr -plot roc prc box polar hp dr dist pie bar -plot_set test

```

### Protein function annotation
```python
python sesica_clf.py -base_dir ../data/go -data_type hetero -bmk_vec_a ../data/go/cc_bmk_vec_a.txt -bmk_vec_b ../data/go/cc_bmk_vec_b.txt -bmk_label ../data/go/pos_label.txt ../data/go/neg_label.txt -clf svm rf ert knn mnb gbdt goss dart mlp -metric aupr -gs_mode 2

python sesica_rank.py -base_dir ../data/go -rank ltr -clf svm rf ert knn mlp -metric aupr -gs_mode 2

python sesica_plot.py -base_dir ../data/go -data_type homo -clf svm rf ert knn mlp -rank ltr -plot polar dr dist pie bar -plot_set test

```

## Deep-Learning Model Reference

| Model | Reference |
| ------------- | ------------- |
| ARC-I | <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>  |
| ARC-II  | <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf">Learning Deep Structured Semantic Models for Web Search using Clickthrough Data</a> |
| CDSSM | <a href="https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/">Learning Semantic Representations Using Convolutional Neural Networks for Web Search</a> |
| DRMM | <a href="https://arxiv.org/abs/1711.08611">A Deep Relevance Matching Model for Ad-hoc Retrieval</a> |
| DRMMTKS | <a href="https://link.springer.com/chapter/10.1007/978-3-030-01012-6_2">A Deep Top-K Relevance Matching Model for Ad-hoc Retrieval</a>. |
| MatchLSTM | <a href="https://arxiv.org/abs/1608.07905">Machine Comprehension Using Match-LSTM and Answer Pointer</a> |
| DUET | <a href="https://dl.acm.org/citation.cfm?id=3052579">Learning to Match Using Local and Distributed Representations of Text for Web Search</a> |
| KNRM | <a href="https://arxiv.org/abs/1706.06613">End-to-End Neural Ad-hoc Ranking with Kernel Pooling</a> |
| ConvKNRM | <a href="http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf">Convolutional neural networks for soft-matching n-grams in ad-hoc search</a> |
| ESIM | <a href="https://arxiv.org/abs/1609.06038">Enhanced LSTM for Natural Language Inference</a> |
| BiMPM | <a href="https://arxiv.org/abs/1702.03814">Bilateral Multi-Perspective Matching for Natural Language Sentences</a> |
| MatchPyramid | <a href="https://arxiv.org/abs/1602.06359">Text Matching as Image Recognition</a> |
| Match-SRNN | <a href="https://arxiv.org/abs/1604.04378">Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN</a> |
| aNMM | <a href="https://arxiv.org/abs/1801.01641">aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model</a> |
| MV-LSTM | <a href="https://arxiv.org/pdf/1511.08277.pdf">A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations</a> |
| DIIN | <a href="https://arxiv.org/pdf/1709.04348.pdf">Natural Lanuguage Inference Over Interaction Space</a> |
| HBMP | <a href="https://arxiv.org/pdf/1808.08762.pdf">Sentence Embeddings in NLI with Iterative Refinement Encoders</a> |



## Code reference
**deep-learning semantic similarity calculation reference**
+ https://github.com/NTMC-Community/MatchZoo-py

**LTR part code reference**
+ https://github.com/microsoft/LightGBM
+ https://github.com/jma127/pyltr
+ https://github.com/slundberg/shap