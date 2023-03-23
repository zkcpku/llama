TARGET_FOLDER=../llama_weight
model_size=7B
MP=1
torchrun --nproc_per_node ${MP} example.py --ckpt_dir ${TARGET_FOLDER}/${model_size} --tokenizer_path ${TARGET_FOLDER}/tokenizer.model
