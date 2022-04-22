# Sesica
semantic similarity calculation

**Software Requirements:**

* [Python3](https://docs.python-guide.org/starting/install3/linux/)
* [virtualenv](https://virtualenv.pypa.io/en/latest/installation/) or [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) (Optional If using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional If using GPU)

Sesica has been tested on Windows, Ubuntu 16.04, and 18.04 operating systems.

## Installation

### virtualenv

```shell
virtualenv -p python3.7 venv

source ./venv/bin/activate

pip install -r requirements.txt
```

### Anaconda

```shell
conda create -n venv python=3.7

conda activate venv

pip install -r requirements.txt
```

## Command

### identification of circRNA-disease associations
```python
python sesica_clf.py -base_dir ../data/iCircDA -data_type hetero -bmk_vec_a ../data/iCircDA/bmk_circRNA.txt -bmk_vec_b ../data/iCircDA/bmk_disease.txt -bmk_label ../data/iCircDA/benchmark_pos.txt ../data/iCircDA/benchmark_neg.txt -clf svm rf ert knn mnb gbdt goss dart mlp -metric auc -gs_mode 2

python sesica_rank.py -base_dir ../data/iCircDA -rank ltr -clf svm rf ert knn mnb gbdt goss dart mlp -metric auc -gs_mode 2

python sesica_plot.py -base_dir ../data/iCircDA -clf svm rf ert knn mnb gbdt goss dart mlp -rank ltr -plot roc prc box polar hp dr dist pie bar -plot_set test
```

### Model

- [DRMM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/drmm.py): this model is an implementation of <a href="http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf">A Deep Relevance Matching Model for Ad-hoc Retrieval</a>.
- [DRMMTKS](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/drmmtks.py): this model is an implementation of <a href="https://link.springer.com/chapter/10.1007/978-3-030-01012-6_2">A Deep Top-K Relevance Matching Model for Ad-hoc Retrieval</a>.
- [ARC-I](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/arci.py): this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>
- [ARC-II](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/arcii.py): this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>
- [DSSM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/dssm.py): this model is an implementation of <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf">Learning Deep Structured Semantic Models for Web Search using Clickthrough Data</a>
- [CDSSM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/cdssm.py): this model is an implementation of <a href="https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/">Learning Semantic Representations Using Convolutional Neural Networks for Web Search</a>
- [MatchLSTM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/matchlstm.py):this model is an implementation of <a href="https://arxiv.org/abs/1608.07905">Machine Comprehension Using Match-LSTM and Answer Pointer</a>
- [DUET](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/duet.py): this model is an implementation of <a href="https://dl.acm.org/citation.cfm?id=3052579">Learning to Match Using Local and Distributed Representations of Text for Web Search</a>
- [KNRM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/knrm.py): this model is an implementation of <a href="https://arxiv.org/abs/1706.06613">End-to-End Neural Ad-hoc Ranking with Kernel Pooling</a>
- [ConvKNRM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/conv_knrm.py): this model is an implementation of <a href="http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf">Convolutional neural networks for soft-matching n-grams in ad-hoc search</a>
- [ESIM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/esim.py): this model is an implementation of <a href="https://arxiv.org/abs/1609.06038">Enhanced LSTM for Natural Language Inference</a>
- [BiMPM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/bimpm.py): this model is an implementation of <a href="https://arxiv.org/abs/1702.03814">Bilateral Multi-Perspective Matching for Natural Language Sentences</a>
- [MatchPyramid](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/match_pyramid.py): this model is an implementation of <a href="https://arxiv.org/abs/1602.06359">Text Matching as Image Recognition</a>
- [Match-SRNN](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/match_srnn.py): this model is an implementation of <a href="https://arxiv.org/abs/1604.04378">Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN</a>
- [aNMM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/anmm.py): this model is an implementation of <a href="https://arxiv.org/abs/1801.01641">aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model</a>
- [MV-LSTM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/mvlstm.py): this model is an implementation of <a href="https://arxiv.org/pdf/1511.08277.pdf">A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations</a>
- [DIIN](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/diin.py): this model is an implementation of <a href="https://arxiv.org/pdf/1709.04348.pdf">Natural Lanuguage Inference Over Interaction Space</a>
- [HBMP](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/hbmp.py): this model is an implementation of <a href="https://arxiv.org/pdf/1808.08762.pdf">Sentence Embeddings in NLI with Iterative Refinement Encoders</a>


## reference
**deep-learning semantic similarity calculation reference**
+ https://github.com/NTMC-Community/MatchZoo-py

**LTR part code reference**
+ https://github.com/microsoft/LightGBM
+ https://github.com/jma127/pyltr
+ https://github.com/slundberg/shap