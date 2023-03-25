TARGET_FOLDER=../llama_weight
model_size=7B
MP=1
torchrun --nproc_per_node ${MP} test_ppl_acc.py --ckpt_dir ${TARGET_FOLDER}/${model_size} --tokenizer_path ${TARGET_FOLDER}/tokenizer.model --test_tmp_bool False


# # sh eval_single.sh
# False
# <class 'bool'>
# > initializing model parallel with size 1
# > initializing ddp with size 1
# > initializing pipeline with size 1
# Loading
# Loaded in 11.47 seconds
# {'loss': [4.158874988555908, 5.460655212402344], 'NextToken_accnum': [3.0, 1.0], 'nonzero_token_num': [8, 7], 'mean_loss': 4.7663726806640625}
# {'loss': 4.766372203826904, 'NextToken_accnum': 4, 'nonzero_token_num': 15}
# {'loss': [3.027263641357422, 4.595463275909424], 'NextToken_accnum': [3.0, 1.0], 'nonzero_token_num': [7, 6], 'mean_loss': 3.7510480880737305}
# {'loss': 3.7510480880737305, 'NextToken_accnum': 4, 'nonzero_token_num': 13}