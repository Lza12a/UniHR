# UniHR

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

The training and testing script for wikidata12k:

```
python -u ./src/run.py --dataset "wikidata12k" --device "3" --vocab_size 13201 --vocab_file "./data/wikidata12k/vocab.txt" --train_file "./data/wikidata12k/train.json" --test_file "./data/wikidata12k/test.json" --ground_truth_file "./data/wikidata12k/all.json" --num_workers 5 --num_relations 26 --num_ent 12554 --max_seq_len 7 --max_arity 4 --hidden_dim 200 --global_layers 1 --global_dropout 0.2 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.2 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 --lr 1e-3 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.2 --hyperedge_dropout 0.1 --epoch 300 --warmup_proportion 0.1 
```

The joint training script for wikidata12k & WikiPeople and testing for WikiPeople:
```
python -u ./src/run.py --dataset "wikimix" --device "1" --vocab_size 44639 --vocab_file "./data/wikimix/vocab.txt" --train_file "./data/wikimix/train+valid.json" --test_file "./data/wikimix/test.json" --ground_truth_file "./data/wikimix/all.json" --num_workers 5 --num_relations 185 --max_seq_len 13 --max_arity 7 --hidden_dim 200 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 --lr 1e-3 --weight_deca 0.01 --entity_soft 0.1 --relation_soft 0.2 --hyperedge_dropout 0.5 --epoch 400 --warmup_proportion 0.1
```

The testing for wikidata12k after joint training
```
python -u ./src/run.py --dataset "wikimix" --device "2" --vocab_size 44639 --vocab_file "./data/wikimix/vocab.txt" --train_file "./data/wikimix/train+valid.json" --test_file "./data/wikidata12k/test.json" --ground_truth_file "./data/wikimix/all.json" --num_workers 5 --num_relations 185 --max_seq_len 13 --max_arity 7 --hidden_dim 200 --global_layers 2 --global_dropout 0.1 --global_activation "elu" --global_heads 4 --local_layers 2 --local_dropout 0.1 --local_heads 4 --decoder_activation "gelu" --batch_size 2048 --lr 1e-3 --weight_deca 0.01 --entity_soft 0.2 --relation_soft 0.2 --hyperedge_dropout 0.5 --epoch 400 --warmup_proportion 0.1 --test_only --ckpt_save_dir "ckpts/wikimix_epoch_400.ckpt" 
```
