# BRUCE - Bundle Recommendation Using Contextualized item Embeddings

This is our official implementation for the paper: BRUCE - Bundle Recommendation Using Contextualized item Embeddings<br/>
This code and repository have been anonymized for peer review.

# Run the Code 
## Best Configurations
The BRUCE architecture is modular and combines several independent 
components which can be configured to best match the
data and task <br/>

Here are the best configurations for running BRUCE on each dataset.
Steam:<br/>
```
Main.py --dataset_string=Steam --description=noPretrain_10k_op_bert --op_after_transformer=bert --num_epochs=10000 --num_transformer_layers=1 --start_val_from=8000
```
Youshu:<br/>
```
Main.py --description=UserBert7000 --num_epochs=7000 --start_val_from=4000 --useUserBert
```
NetEase:<br/>

1. Create data for pretraining embeddings (too heavy to put in github) using DataAndResultsAnalysis/CreateItemItemDataForPretrain.ipynb
2. Pretrain Embeddings:
```
Pretrain/PreTrainer.py --dataset_string=NetEase --description=ready --run_pretrain --lr=1e-3 --weight_decay=1e-5 --evaluate_every=500 --start_val_from=-1 --num_epochs=50 --pretrain_train_batch_size=2048
```
3. Train model - divided to 2 steps:
```
a. Main.py --dataset_string=NetEase --seed=111 --description=pretrained_userBert_5k_2048 --useUserBertV2 --num_epochs=5000 --dont_test_last_epoch --batch_size=2048 --start_val_from=4000 --use_pretrained --pretrained_path=PretainedOutputDir/Pretrain/best_model.pth 
b. Main.py --dataset_string=NetEase --seed=111 --description=pretrained_userBert_2k_after_5k_2048 --useUserBertV2 --num_epochs=2000 --batch_size=2048 --evaluate_every=500 --num_epochs_on_previous_train=10 --start_val_from=-1 --model_path=OutputDirOf2a/latest_model.pth 
```

### BRUCE Configurations
BRUCE code is modular and can be used and changed according to the need and task.
#### 1. Using pretrained embeddings
The default configuration is to randomly initialize item embeddings. <br>
In order to use pretrained embeddings you need to do the following steps. <br>
a. Create the data for the pretraining the embeddings - using the code on DataAndResultsAnalysis/CreateItemItemDataForPretrain.ipynb <br>
b. Pretrain the embeddings - by running Pretrain/PreTrainer.py (the best pretrain arguments per dataset are commented in the file). <br>
c. Train with pretrained embeddings: by adding the parameters --use_pretrained --pretrained_path=PretainedOutputDir/Pretrain/best_model.pth

#### 2. Integrating User Information
The following user integration techniques are supported: <br>
a. Concatenation of the user to each item. default option, you also need to pass the op_after_transformer elaborated in the next section. <br>
The models' code is under the PreUL dir. <br>
b. User first. by passing --useUserBert or --useUserBertV2 (the first shares the Transformer layer with the auxiliary task of items recommendation while the second does not). <br>
The models' code is under the UserBert dir. <br>
c. Post Transformer Integration. by passing --usePostUL,  you also need to pass the op_after_transformer elaborated in the next section. <br>
The models' code is under the PostUL dir. <br>

#### 3. Aggregation Methods
The aggregation method preformed after the Transformer layer, the following are supported: <br>
a. Concatenation. --op_after_transformer=concat<br>
b. Summation --op_after_transformer=sum<br>
c. Averaging --op_after_transformer=avg<br>
d. First item (BERT-like) aggregation --op_after_transformer=bert<br>
e. Bundle embedding BERT-like aggregation --bundleEmbeddings --op_after_transformer=bert


#### 4. Multi-task Learning
You can avoid the multi-task learning process by using the --dont_multi_task flag.


# Citation
If you use this code, please cite our paper. Thanks!
```
@inproceedings{
  author    = {Anonymized for review},
  title     = {BRUCE - Bundle Recommendation Using Contextualized item Embeddings},
  year      = {2022}
}

```

### Acknowledgements
Portions of this code are based on the [BGCN paper's code](https://github.com/cjx0525/BGCN) and the [DAM paper's code](https://github.com/yliuSYSU/DAM).`
