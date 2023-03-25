TARGET_FOLDER=/code/llama_weight
model_size=7B
MP=1

CORPUS_PATH=/code/llama/CSN_python_func_code_string_top1000_norm.json
OUTPUT_PATH=/code/llama/CSN_python_func_code_string_top1000_norm.save.json
# OUTPUT_PATH=/model/CSN_python_func_code_string_top1000.save.json


torchrun --nproc_per_node ${MP} test_ppl_acc_for_corpus.py --ckpt_dir ${TARGET_FOLDER}/${model_size} --tokenizer_path ${TARGET_FOLDER}/tokenizer.model --max_seq_len 2048 --max_batch_size 8 --corpus_path ${CORPUS_PATH} --output_path ${OUTPUT_PATH} --not_consider_eos True 
# def main(
#     ckpt_dir: str,
#     tokenizer_path: str,
#     max_seq_len: int = 512,
#     max_batch_size: int = 32,
#     corpus_path: str = None,
#     output_path: str = None,
#     not_consider_eos: bool = True):
