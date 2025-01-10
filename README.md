# Implementation of HiTIN: (Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification) on Custom Dataset.

Custom Dataset Implementation for the paper "HiTIN: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification" . [[arXiv](https://arxiv.org/abs/2305.15182)][[pdf](https://arxiv.org/pdf/2305.15182.pdf)][[bilibili](https://www.bilibili.com/video/BV1vL411i7uY/?share_source=copy_web&vd_source=a9cc6ff9a8cf3c92bf2375da5b56a007)]
1. This method specifically encodes the text using BERT based text encoder, then further the label graph is encoded using TreeEncoders. This is helpful in capturing the hierarchial information and passing the same to the model.
2. The method requires the labels to be a DAG (Directed Acyclic Graph).
3. In the custom dataset we have 3 classes cat1,cat2,cat3, out of them only cat1 and cat2 are mutually exclusive while cat3 is not i.e. Suppose for cat2_lbl1 and cat2_lbl2, there can be a possibility that cat3_lbl1 can be a subclass of both of them at different times.
4. So we have combined the cat3 and cat2 classes and have hence generated the taxonomy for the same. 
5. General data exploration for the data has been included in the data_exploration.ipynb notebook.


## Requirements
You can create the environemnt using the yml file and the command given below.
```shell
conda env create -f environment.yml
```
Here are the basic env requirements
- Python == 3.7.13
- numpy == 1.21.5
- torch == 1.11.0
- scikit-learn == 1.0.2
- transformers==4.19.2
- numba==0.56.2
- glove.6B.300d

## Data preparation

```shell
cd data
python preprocess_custom.py
```

```shell
python hiearchy_tree_statistic.py ./config/tin-custom-data.json
```



### Conduct experiments on your own data

An additional step is required to count the prior probabilities between parent and child labels by running `python hiearchy_tree_statistic.py your_config_file_path`. HiTIN only requires unweighted adjacency matrix of label hierarchies but they still retain this property and save the statistics in `data/DATASET_prob.json` as they also implement baseline methods including TextRCNN, BERT-based HiAGM. 

The final dataset should be organied in the given below format

```
{
    "doc_label": ["Computer", "MachineLearning", "DeepLearning", "Neuro", "ComputationalNeuro"],
    "doc_token": ["I", "love", "deep", "learning"],
    "doc_keyword": ["deep learning"],
    "doc_topic": ["AI", "Machine learning"]
}

where "doc_keyword" and "doc_topic" are optional.
```

then, replace the label name with your dataset's in line143~146 of `helper/hierarchy_tree_statistics.py` and run:

```shell
python hierarchy_tree_statistic.py your_config_file_path
```


## Train


```
python train.py [-h] -cfg CONFIG_FILE [-b BATCH_SIZE] [-lr LEARNING_RATE]
                [-l2 L2RATE] [-p] [-k TREE_DEPTH] [-lm NUM_MLP_LAYERS]
                [-hd HIDDEN_DIM] [-fd FINAL_DROPOUT] [-tp {root,sum,avg,max}]
                [-hp HIERAR_PENALTY] [-ct CLASSIFICATION_THRESHOLD]
                [--log_dir LOG_DIR] [--ckpt_dir CKPT_DIR]
                [--begin_time BEGIN_TIME]

optional arguments:
  -h, --help            show this help message and exit
  -cfg CONFIG_FILE, --config_file CONFIG_FILE
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        input batch size for training (default: 32)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  -l2 L2RATE, --l2rate L2RATE
                        L2 penalty lambda (default: 0.01)
  -p, --load_pretrained
  -k TREE_DEPTH, --tree_depth TREE_DEPTH
                        The depth of coding tree to be constructed by CIRCA
                        (default: 2)
  -lm NUM_MLP_LAYERS, --num_mlp_layers NUM_MLP_LAYERS
                        Number of layers for MLP EXCLUDING the input one
                        (default: 2). 1 means linear model.
  -hd HIDDEN_DIM, --hidden_dim HIDDEN_DIM
                        Number of hidden units for HiTIN layer (default: 512)
  -fd FINAL_DROPOUT, --final_dropout FINAL_DROPOUT
                        Dropout rate for HiTIN layer (default: 0.5)
  -tp {root,sum,avg,max}, --tree_pooling_type {root,sum,avg,max}
                        Pool strategy for the whole tree in Eq.11. Could be
                        chosen from {root, sum, avg, max}.
  -hp HIERAR_PENALTY, --hierar_penalty HIERAR_PENALTY
                        The weight for L^R in Eq.14 (default: 1e-6).
  -ct CLASSIFICATION_THRESHOLD, --classification_threshold CLASSIFICATION_THRESHOLD
                        Threshold of binary classification. (default: 0.5)
  --log_dir LOG_DIR     Path to save log files (default: log).
  --ckpt_dir CKPT_DIR   Path to save checkpoints (default: ckpt).
  --begin_time BEGIN_TIME
                        The beginning time of a run, which prefixes the name
                        of log files.
```

The config file for this is present in `./config`. 

**Before running, the last thing to do is modify the `YOUR_DATA_DIR`, `YOUR_BERT_DIR` in the json file.**


An example of training HiTIN on custom dataset with **BERT** as the text encoder:

```shell
python train.py -cfg config/tin-custom-data.json -k 2 -b 12 -hd 768 -lr 1e-4 -tp sum
```

Best Macro F1 Model: https://1drv.ms/u/s!AlZ2frx161IqgdRwz9m3Ub5k15_pdw?e=C0ghYF
Best Micro F1 Model: https://1drv.ms/u/s!AlZ2frx161IqgdRyLr_q00XSOhledA?e=Vs01i5
Glove Embeddings: https://1drv.ms/t/s!AlZ2frx161IqgdRvFiKijt3A_l9zVQ?e=u5Dnyn
BERT Base: https://1drv.ms/u/s!AlZ2frx161IqgdRxezqcGIvp2IQvTQ?e=qzb3A6
