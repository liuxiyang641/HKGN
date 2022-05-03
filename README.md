# Heterogeneous Graph Neural Network with Hypernetworks for Knowledge Graph Embedding

## Overview
The source code for **HKGN: Heterogeneous Graph Neural Network with Hypernetworks for Knowledge Graph Embedding**.

```
├── model
│   ├── __init__.py
│   ├── encoder_decoder.py    The overall HKGN model.
│   ├── gcn_encoder.py        HGNN encoder.
│   └── hyper_conv_layer.py   Multi-relational graph convolution.
└── run.py  Root script for running the project.
```
## Data preprocessing

Unzip the compressed datasets.
```bash
mkdir data
unzip data_compressed/FB15k-237.zip -d data/
unzip data_compressed/WN18RR.zip -d data/
```

Then you will get all the essential data to reproduce the results reported in the paper.

**FB15k-237**

```
data
├── FB15k-237
Original data:
│   └── train.txt
│   ├── valid.txt
│   ├── test.txt
Subsets divided by relation categories:
│   ├── 1-1.txt
│   ├── 1-n.txt
│   ├── n-1.txt
│   ├── n-n.txt
Subsets divided by entity degrees:
│   ├── ent100.txt  [0, 100)
│   ├── ent200.txt  [100, 200)
│   ├── ent300.txt  [200, 300)
│   ├── ent400.txt  [300, 400)
│   ├── ent500.txt  [400, 500)
│   ├── ent1000.txt [500, 1000)
│   ├── entmax.txt  [1000, max)
```

**WN18RR**

```
├──WN18RR
Original data:
    └── train.txt
    ├── valid.txt
    ├── test.txt
Subsets divided by entity degrees:
    ├── ent10.txt  [0, 10)
    ├── ent25.txt  [10, 25)
    ├── ent50.txt  [25, 50)
    ├── ent100.txt [50, 100)
    ├── ent500.txt [100, max)
```

## Dependencies

- Python 3.x
- torch-scatter (make sure to be compatible with your own pytorch version, please refer to [torch-scatter](https://github.com/rusty1s/pytorch_scatter))
- tqdm
- PyTorch >= 1.5.0
- [ordered-set](https://pypi.org/project/ordered-set/)
- numpy

Dependencies can be installed using `requirements.txt`. 

## Training the model

To reproduce the best performance we have found:

```bash
# default dataset: FB15k-237
# FB25k-237 layer 2
python run.py -name test_fb_layer2 -gcn_drop 0.4 -model hyper_gcn -gpu 0 -exp hyper_mr_parallel -gcn_layer 2 -layer2_drop 0.2 -layer1_drop 0.3
# FB15k-237 layer 1
python run.py -name test_fb_layer1 -gcn_drop 0.4 -model hyper_gcn -gpu 0 -exp hyper_mr_parallel
# WN18RR layer 1
python run.py -name test_wn18rr_layer1 -gcn_drop 0.4 -model hyper_gcn -batch 256 -gpu 0 -data WN18RR
```

The detailed hyperparameters:

|        Hyperparameters        |          FB15k-237          | WN18RR |
| :---------------------------: | :-------------------------: | :----: |
|             $d_x$             |             100             |  100   |
|             $d_y$             |              2              |   2    |
|             $d_z$             |             100             |  100   |
|     Relaional kernel size     |             3x3             |  3x3   |
|  Number of Relaional filters  | 32 (layer 1) / 16 (layer 2) |   32   |
|          HKGN layers          |              2              |   1    |
| Initial entity embedding size |             100             |  100   |
|  Final entity embedding size  |             200             |  200   |
|        Message dropout        |             0.4             |  0.4   |
|          Batch size           |            1024             |  256   |
|         Learning rate         |            0.001            | 0.001  |
|        Lable smoothing        |             0.1             |  0.1   |

Customize the training strategies (**parallel** and **iterative**) for multi-relational knowledge graphs by setting the argument `-exp`: 

- `hyper_mr_parallel`: Performing all single-relational graph convolution simultaneously (default). High GPU memory footprints and quick training speed.
- `hyper_mr_iter`: Performing different single-relational graph convolution iteratively. Lower GPU memory requirements and slower training speed.

If you got out of memory error, try to clear cached memory by passing the argument: `-empty_gpu_cache`.

The learned model will be automatically saved in directory `/checkpoints`, pass the argument `-restore` to resume your saved model, e.g.:

```bash
-restore -name your_saved_model_name
```

Choose specific data file as the test set by setting `-test_data`.

```bash
-test_data test(defalut)/1-n/n-1/.../ent100/..
```

## Distribution statistics

Statistics of different relation categories on FB15k-237:

|             | 1-1  |  1-N  |  N-1  |  N-N   |
| :---------: | :--: | :---: | :---: | :----: |
| #Relations  |  17  |  26   |  86   |  108   |
|  #Training  | 4278 | 12536 | 50635 | 204666 |
| #Validation | 167  | 1043  | 3936  | 12389  |
|    #Test    | 192  | 1293  | 4508  | 14473  |

Statistics of different degree scopes on FB15k-237:

| Enitity  degree scopes | #Entities | #Test |
| :--------------------: | :-------: | :---: |
|        [0, 100)        |   13839   | 16385 |
|       [100, 200)       |    496    | 2055  |
|       [200, 300)       |    76     |  525  |
|       [300, 400)       |    44     |  493  |
|      [400, 5000)       |    35     |  389  |
|      [500, 1000)       |    35     |  373  |
|      [1000, max)       |    16     |  246  |

Statistics of different degree scopes on WN18RR:

| Enitity  degree scopes | #Entities | #Test |
| :--------------------: | :-------: | :---: |
|        [0, 10)         |   38102   | 2595  |
|        [10, 25)        |   2497    |  417  |
|        [25, 50)        |    243    |  47   |
|       [50, 100)        |    65     |  29   |
|       [100, 500)       |    36     |  46   |
