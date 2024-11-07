# UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction

### Introduction
Beyond-triple fact representations including hyper-relational facts with auxiliary key-value pairs, temporal facts with additional timestamps, and nested facts implying relationships between facts, are gaining significant attention. However, existing link prediction models are usually designed for one specific type of facts, making it difficult to generalize to other fact representations. To overcome this
limitation, we propose a Unified Hierarchical Representation learning framework (UniHR)
for unified knowledge graph link prediction. It consists of a unified Hierarchical Data Repre-
sentation (HiDR) module and a unified Hierarchical Representation Learning (HiRL) module as graph encoder. The HiDR module unifies hyper-relational KGs, temporal KGs, and nested factual KGs into triple-based representations. Then HiRL incorporates intra-fact and inter-fact message passing, focusing on enhanc-
ing the semantic information within individual facts and enriching the structural information between facts. Experimental results demonstrate that our UniHR outperforms baselines designed for one specific kind of KG, indicating strong generalization capability of HiDR form and the effectiveness of HiRL module.

### 🔎 Requirements
- `PyTorch 1.10.0`
- `torch-scatter  2.0.9`
- `torch-sparse 0.6.13`
- `torch-cluster 1.6.0`
- `torch-geometric 2.1.0.post1`
- `numpy 1.23.3`

All the experiments are conducted on a single 80G NVIDIA A800.

#### Setup with Conda

```
bash env.sh
```

### 🚀 Running

The training and testing script for WikiPeople:

```
python -u ./src/run.py --dataset "wikipeople" --device "0" --vocab_size 35005 --vocab_file "./data/wikipeople/vocab.txt" --train_file "./data/wikipeople/train+valid.json" --test_file "./data/wikipeople/test.json" --ground_truth_file "./data/wikipeople/all.json" --num_workers 10 --num_relations 178 --max_seq_len 13 --max_arity 7 --hidden_dim 200 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 --lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.2 --hyperedge_dropout 0.5 --epoch 300 --warmup_proportion 0.1
```

The training and testing script for WD50K:

```
python -u ./src/run.py --dataset "wd50k" --device "1" --vocab_size 47688 --vocab_file "./data/wd50k/vocab.txt" --train_file "./data/wd50k/train+valid.json" --test_file "./data/wd50k/test.json" --ground_truth_file "./data/wd50k/all.json" --num_workers 10 --num_relations 531 --max_seq_len 19 --max_arity 10 --hidden_dim 200 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 --lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.1 --hyperedge_dropout 0.1 --epoch 300 --warmup_proportion 0.1 
```

The training and testing script for DBHE_atomic:

```
python -u ./src/run.py --dataset "DBHE" --device "1" --vocab_size 67173 --vocab_file "./data/DBHE/vocab.txt" --train_file "./data/DBHE/train+aug.json" --test_file "./data/DBHE/test.json" --ground_truth_file "./data/DBHE/all.json" --num_workers 10 --num_relations 95 --max_seq_len 3 --max_arity 2 --hidden_dim 200 --global_layers 2 --global_dropout 0.3 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 --lr 5e-4 --weight_deca 0.01 --entity_soft 0.3 --relation_soft 0.3 --hyperedge_dropout 0.0 --epoch 250 --warmup_proportion 0.1
```


The training and testing script for DBHE_nest:

```
python -u ./src/run.py --nest_meta True --ckpt "ckpts/DBHE_epoch_250.ckpt" --dataset "DBHE" --device "1" --vocab_size 67173 --vocab_file "./data/DBHE/vocab.txt" --train_file "./data/DBHE/meta_train+valid.json" --test_file "./data/DBHE/meta_test.json" --ground_truth_file "./data/DBHE/meta_all.json" --num_workers 10 --num_relations 95 --max_seq_len 3 --max_arity 2 --hidden_dim 200 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 --lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.2 --hyperedge_dropout 0.0 --epoch 1000 --warmup_proportion 0.1
```


The training and testing script for FBHE_atomic:

```
python -u ./src/run.py --dataset "FBHE" --device "0" --vocab_size 262884 --vocab_file "./data/FBHE/vocab.txt" --train_file "./data/FBHE/train+aug+valid.json" --test_file "./data/FBHE/test.json" --ground_truth_file "./data/FBHE/all.json" --num_workers 10 --num_relations 247 --max_seq_len 3 --max_arity 2 --hidden_dim 200 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 --lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.2 --hyperedge_dropout 0.0 --epoch 300 --warmup_proportion 0.1

```

The training and testing script for FBHE_nest:

```
python -u ./src/run.py --nest_meta True --ckpt "ckpts/FBHE_epoch_299.ckpt" --dataset "FBHE" --device "0" --vocab_size 262884 --vocab_file "./data/FBHE/vocab.txt" --train_file "./data/FBHE/meta_train+valid.json" --test_file "./data/FBHE/meta_test.json" --ground_truth_file "./data/FBHE/meta_all.json" --num_workers 10 --num_relations 247 --max_seq_len 3 --max_arity 2 --hidden_dim 200 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 1024 --lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.2 --hyperedge_dropout 0.0 --epoch 200 --warmup_proportion 0.1
```

The training and testing script for FBH_atomic:

```
python -u ./src/run.py --dataset "FBH" --device "1" --vocab_size 262880 --vocab_file "./data/FBH/vocab.txt" --train_file "./data/FBH/train+aug.json" --test_file "./data/FBH/test.json" --ground_truth_file "./data/FBH/all.json" --num_workers 10 --num_relations 243 --max_seq_len 3 --max_arity 2 --hidden_dim 200 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 --lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.2 --hyperedge_dropout 0.0 --epoch 300 --warmup_proportion 0.1

```

The training and testing script for FBH_nest:

```
python -u ./src/run.py --nest_meta True --ckpt "ckpts/FBHE_epoch_299.ckpt" --dataset "FBH" --device "5" --vocab_size 262880 --vocab_file "./data/FBH/vocab.txt" --train_file "./data/FBH/meta_train+valid.json" --test_file "./data/FBH/meta_test.json" --ground_truth_file "./data/FBH/meta_all.json" --num_workers 10 --num_relations 247 --max_seq_len 3 --max_arity 2 --hidden_dim 200 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 --lr 5e-4 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.2 --hyperedge_dropout 0.0 --epoch 200 --warmup_proportion 0.1
```


The training and testing script for wikidata12k:

```
python -u ./src/run.py --dataset "wikidata12k" --device "3" --vocab_size 13201 --vocab_file "./data/wikidata12k/vocab.txt" --train_file "./data/wikidata12k/train.json" --test_file "./data/wikidata12k/test.json" --ground_truth_file "./data/wikidata12k/all.json" --num_workers 5 --num_relations 26 --num_ent 12554 --max_seq_len 7 --max_arity 4 --hidden_dim 200 --global_layers 1 --global_dropout 0.2 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.2 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 --lr 5e-4 --weight_deca 0.01 --entity_soft 0.4 --relation_soft 0.3 --hyperedge_dropout 0.0 --epoch 300 --warmup_proportion 0.1 
```
