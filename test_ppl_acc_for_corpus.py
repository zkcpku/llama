from test_ppl_acc import *

import sys
import os
import json
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_jsonl(filename):
    with open(filename, 'r') as f:
        count = 0
        for line in f:
            count += 1
    return count

def read_jsonl(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield json.loads(line)
def write_jsonl(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')
def write_jsonl_append(filename, each_data):
    with open(filename, 'a+') as f:
        f.write(json.dumps(each_data) + '\n')

def read_jsonl_batch(filename, batch_size):
    with open(filename, 'r') as f:
        batch = []
        for line in f:
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    corpus_path: str = None,
    output_path: str = None,
    not_consider_eos: bool = True):
    
    # print params
    print_params = {
        'ckpt_dir': ckpt_dir,
        'tokenizer_path': tokenizer_path,
        'max_seq_len': max_seq_len,
        'max_batch_size': max_batch_size,
        'corpus_path': corpus_path,
        'output_path': output_path,
        'not_consider_eos': not_consider_eos,
    }
    logger.info("Params: %s", print_params)

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    assert corpus_path is not None
    assert output_path is not None

    json_reader_batch = read_jsonl_batch(corpus_path, max_batch_size)
    total_batch = count_jsonl(corpus_path) // max_batch_size

    for each_data in tqdm(json_reader_batch, total=total_batch):
        prompts = [each['text'] for each in each_data]
        ids = [each['id'] for each in each_data]
        results = generator.eval(
            prompts, max_gen_len=max_seq_len, return_each_sample=True, not_consider_eos = not_consider_eos
        )
        results['ids'] = ids
        write_jsonl_append(output_path, results)

if __name__ == "__main__":
    fire.Fire(main)

