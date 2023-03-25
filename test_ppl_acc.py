# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMAEval


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMAEval:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMAEval(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator




def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    test_tmp_bool: bool = True,
):
    print(test_tmp_bool)
    print(type(test_tmp_bool))
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

#     prompts = [
#         "I believe the meaning of life is",
#     "I believe the meaning of life"]
    prompts = [
        "I believe the meaning of life is",
    "I believe a meaning of life"]
    
#     prompts = [
#         "I believe the meaning of life is"]
    
#     prompts = [
#         "I believe the meaning of life is"]
    
    
    results = generator.eval(
        prompts, max_gen_len=256, return_each_sample=True, not_consider_eos = False
    )
    print(results)
    
    results = generator.eval(
        prompts, max_gen_len=256, return_each_sample=False, not_consider_eos = False
    )
    print(results)
    
    results = generator.eval(
        prompts, max_gen_len=256, return_each_sample=True, not_consider_eos = True
    )
    print(results)

    
    
    results = generator.eval(
        prompts, max_gen_len=256, return_each_sample=False, not_consider_eos = True
    )
    print(results)
#     for result in results:
#         print(result)
#         print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
