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

## reference
**deep-learning semantic similarity calculation reference**
+ https://github.com/NTMC-Community/MatchZoo-py

**LTR part code reference**
+ https://github.com/microsoft/LightGBM
+ https://github.com/jma127/pyltr
+ https://github.com/slundberg/shap