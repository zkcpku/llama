# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer
from torch.nn import CrossEntropyLoss

class LLaMAEval:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_id = self.tokenizer.eos_id


    def eval(
        self,
        prompts: List[str],
        max_gen_len: int,
        return_each_sample = False,
        not_consider_eos = False
    ):
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        if not_consider_eos:
            prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        else:
            prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=True) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            each_len = min(len(t), total_len)
            tokens[k, : each_len] = torch.tensor(t[:each_len]).long()
#         tokenizer = self.tokenizer
#         import ipdb;ipdb.set_trace()
        input_text_mask = tokens != self.tokenizer.pad_id # pad token不参与loss计算 # pad_token == -1
        logits = self.model.forward_logits(tokens, 0)
        
        labels = tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        shift_tokens = tokens[..., :-1].contiguous()
        if not_consider_eos:
            label_mask = shift_labels != self.tokenizer.pad_id
        else:
            label_mask = shift_tokens != self.tokenizer.eos_id
        # label_mask = shift_labels != self.tokenizer.pad_id

        if return_each_sample:
            loss_fct = CrossEntropyLoss(reduction='none')
#             print(shift_logits.view(-1, shift_logits.size(-1)))
#             print(shift_labels)
            each_sample_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            each_sample_loss = each_sample_loss.view(bsz, -1)
            
            
            each_sample_loss = each_sample_loss * label_mask.float()
            each_sample_loss = each_sample_loss.view(bsz, -1)
            each_sample_loss = each_sample_loss.sum(dim=1)

            mean_loss = each_sample_loss.sum() / label_mask.sum()
            mean_loss = mean_loss.item()

            each_sample_loss = each_sample_loss / label_mask.sum(dim=1).float()
            each_sample_loss = each_sample_loss.tolist()

            

            next_token_acc_num = shift_logits.view(-1, shift_logits.size(-1)).argmax(dim=1) == shift_labels.view(-1)
            next_token_acc_num = next_token_acc_num.view(bsz, -1)
            next_token_acc_num = next_token_acc_num * label_mask.float()
            next_token_acc_num = next_token_acc_num.sum(dim=1)
            next_token_acc_num = next_token_acc_num.tolist()

            nonzero_token_num = label_mask.view(bsz, -1)
            nonzero_token_num = nonzero_token_num.sum(dim=1)
            nonzero_token_num = nonzero_token_num.tolist()

            return {
                "loss": each_sample_loss,
                "NextToken_accnum": next_token_acc_num,
                "nonzero_token_num": nonzero_token_num,
                "mean_loss": mean_loss,
            }
        
        loss_fct = CrossEntropyLoss()
        if not_consider_eos:
            flatten_shift_loss_mask = input_text_mask[..., 1:].contiguous().view(-1)
        else:
            flatten_shift_loss_mask = input_text_mask[..., :-1].contiguous().view(-1)
        ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
        eval_loss = loss.mean().item() # 平均所有位置上的loss

        next_token_acc_num = shift_logits.view(-1, shift_logits.size(-1))[ids].argmax(dim=1) == shift_labels.view(-1)[ids]
        nonzero_token_num = shift_labels.view(-1)[ids].size(0)
        
        loss = eval_loss
#         perplexity = torch.exp(torch.tensor(eval_loss))
        
        return {
            "loss": eval_loss,
            "NextToken_accnum": next_token_acc_num.sum().item(),
            "nonzero_token_num": nonzero_token_num, 
        }
    

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
#         import ipdb;ipdb.set_trace()
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            inp_tensor = tokens[:, prev_pos:cur_pos]
#             import ipdb;ipdb.set_trace()
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
